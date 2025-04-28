import argparse
import pandas as pd
import requests
import json
import re
from datasets import load_dataset
from tqdm import tqdm
import logging
import time

# Set up logging to record problematic cases
logging.basicConfig(filename="logs/extract_answer.log", level=logging.INFO)

def parse_arguments():
    """
    Parse command-line arguments for the inference script.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Generate and revise responses using OpenAI models.")
    parser.add_argument("--model_name", type=str, default="openai/chatgpt-4o-latest", help="Model name to use")
    parser.add_argument("--dataset", type=str, default="juletxara/mgsm", help="Dataset name to use")
    parser.add_argument("--max_waits", type=int, default=3, help="Maximum number of retries for revisions")
    parser.add_argument("--num_questions", type=int, default=50, help="Number of questions to process per language")
    parser.add_argument("--languages", nargs="+", default=["bn", "te", "th", "es", "en", "ja"], help="List of languages")
    parser.add_argument("--results_dir", type=str, default="results_gpt/", help="Directory to save results")
    parser.add_argument("--api_key_file", type=str, default="open_router_api.txt", help="File containing API key")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="Tolerance for answer comparison")
    parser.add_argument("--save_tables", action="store_true", help="Save correctness and incorrectness tables as CSV")
    parser.add_argument("--display_tables", action="store_true", help="Display the correctness and incorrectness tables")
    return parser.parse_args()

args = parse_arguments()

# Load the dataset for each specified language
datasets = {
    lang: load_dataset(args.dataset, lang)["test"]
    for lang in args.languages
}

# Read the API key from file
with open(args.api_key_file, "r") as file:
    api_key = file.read().strip()


def extract_answer(text):
    """
    Extract a numerical answer from the given text output, 
    attempting to handle boxed formats and free-form numbers.

    Args:
        text (str): The generated model output.

    Returns:
        float or None: The extracted number, or None if extraction fails.
    """
    try:
        match = re.search(r'\\boxed\{.*?([\d,\.]+).*?\}', text)
        if match:
            number_str = match.group(1)
            number_str = re.sub(r'[^\d\.,-]', '', number_str)
            number_str = number_str.replace(",", "").replace("!", "")
            try:
                return float(number_str)
            except ValueError:
                logging.info(f"Error converting to float: '{number_str}' from text: {text}")
                return None

        numbers = re.findall(r'[-+]?\d{1,3}(?:[,\.]\d{3})*(?:\.\d+)?|\d+', text)
        if numbers:
            number_str = re.sub(r'[^\d.-]', '', numbers[-1])
            try:
                return float(number_str)
            except ValueError:
                logging.info(f"Error converting to float: '{number_str}' from text: {text}")
                return None

    except Exception as e:
        logging.info(f"Error extracting answer from text: {text}\nException: {e}")
        return None

    logging.info(f"No valid answer found in text: {text}")
    return None

def generate_responses():
    """
    Generate model responses for a set of math questions across multiple languages.

    Returns:
        pd.DataFrame: DataFrame containing questions, answers, generated outputs, and correctness labels.
    """
    rows = []
    total_iterations = args.num_questions * len(args.languages)

    with tqdm(total=total_iterations, desc="Generating Responses") as pbar:
        for lang, dataset in datasets.items():
            for i in range(args.num_questions):
                test_sample = dataset[i]
                question = test_sample['question']
                answer_number = test_sample['answer_number']

                # Construct the Chain-of-Thought (CoT) prompt
                prompt = f"""
                Given the following question:
                Question: {question}

                Let's think step by step to find the CoT solution and provide the final answer in the boxed format.
                """

                retry_count = 0
                while retry_count <= args.max_waits:
                    try:
                        response = requests.post(
                            url="https://openrouter.ai/api/v1/chat/completions",
                            headers={
                                "Authorization": f"Bearer {api_key}",
                                "Content-Type": "application/json",
                            },
                            data=json.dumps({
                                "model": args.model_name,
                                "messages": [{"role": "user", "content": prompt}],
                            })
                        )

                        if response.status_code == 200:
                            response_data = response.json()
                            if 'choices' in response_data and len(response_data['choices']) > 0:
                                message_content = response_data['choices'][0]['message']['content']
                                gen_answer = extract_answer(message_content)
                                is_correct = abs(answer_number - gen_answer) < args.tolerance if pd.notnull(gen_answer) else False

                                row = {
                                    "q_no": i + 1,
                                    "lang": lang,
                                    "question": question,
                                    "answer_number": answer_number,
                                    "gen_text": message_content,
                                    "gen_answer": gen_answer,
                                    "is_correct": is_correct
                                }
                                rows.append(row)
                                break

                        else:
                            # Handle server errors with exponential backoff
                            if response.status_code == 524 or response.status_code >= 500:
                                retry_count += 1
                                wait_time = 2 ** retry_count
                                logging.error(f"Error {response.status_code}: Retrying in {wait_time}s")
                                time.sleep(wait_time)
                            else:
                                logging.error(f"Unexpected status code {response.status_code}: {response.text}")
                                break

                    except Exception as e:
                        logging.error(f"Exception during API call: {str(e)}")
                        retry_count += 1
                        time.sleep(2 ** retry_count)

                pbar.update(1)

    return pd.DataFrame(rows)

def generate_revisions(df):
    """
    Generate self-revisions for initially incorrect answers.

    Args:
        df (pd.DataFrame): DataFrame containing original generation results.

    Returns:
        pd.DataFrame: DataFrame updated with self-revision results and wait counts.
    """
    negative_prompt = """
    Solve the following question:
    Question: {question}

    The previous reasoning and answer were incorrect:
    {reasoning_answer}

    Wait... Let me think. There seems to be a mistake in the previous logic or calculation.
    I'll reassess the problem from scratch and provide a new solution step by step,
    carefully avoiding the previous mistake.

    Final answer should be an integer and presented in boxed format.
    """

    for idx, row in tqdm(df[df["is_correct"] == False].iterrows(), desc="Generating Revisions"):
        question = row["question"]
        reasoning_answer = row["gen_text"]
        answer_number = row["answer_number"]
        wait_count = 0
        accumulated_responses = []

        while wait_count < args.max_waits:
            revision_prompt = negative_prompt.format(question=question, reasoning_answer=reasoning_answer)
            prompt = [{"role": "user", "content": revision_prompt}]

            try:
                response = requests.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    data=json.dumps({"model": args.model_name, "messages": prompt})
                )

                if response.status_code == 200:
                    message_content = response.json()['choices'][0]['message']['content']
                    accumulated_responses.append(message_content)

                    revised_answer = extract_answer(message_content)
                    if revised_answer == float(answer_number):
                        break
                    else:
                        accumulated_responses.append("Wait... Let me think again.")
                        wait_count += 1
                        reasoning_answer = message_content
                else:
                    print(f"Error {response.status_code}: {response.text}")
                    wait_count += 1

            except Exception as e:
                logging.info(f"Error generating revision for index {idx}: {e}")
                wait_count += 1

        df.at[idx, "self_revision"] = "\n---\n".join(accumulated_responses)
        df.at[idx, "wait_counts"] = wait_count

    return df

def extract_revisions(df):
    """
    Extract and evaluate the revised answers from self-revision outputs.

    Args:
        df (pd.DataFrame): DataFrame after revision generation.

    Returns:
        pd.DataFrame: DataFrame with additional columns for revised answers and correctness.
    """
    if df["is_correct"].all():
        print("All answers are correct. No revisions needed.")
        df["revised_answer"] = None
        df["is_correct_rev"] = True
        df["wait_counts"] = None
        df["self_revision"] = None
        return df

    revised_answers = []
    is_correct_revs = []
    wait_counts = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Revisions"):
        if row["is_correct"]:
            revised_answers.append(None)
            is_correct_revs.append(True)
            wait_counts.append(None)
            continue

        response = row.get("self_revision", None)
        answer_number = row["answer_number"]

        revised_number = extract_answer(response) if isinstance(response, str) else None
        correct = (abs(revised_number - float(answer_number)) < args.tolerance) if revised_number is not None else False

        revised_answers.append(revised_number)
        is_correct_revs.append(correct)
        wait_counts.append(row.get("wait_counts", None))

    df["revised_answer"] = revised_answers
    df["is_correct_rev"] = is_correct_revs
    df["wait_counts"] = wait_counts
    return df

def transform_comparison_stats(comparison_stats):
    """
    Transform aggregated comparison statistics into correctness and incorrectness tables.

    Args:
        comparison_stats (pd.DataFrame): Aggregated DataFrame with initial and revised correctness counts.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Correctness table and incorrectness table.
    """
    total_initial_correct = comparison_stats['original_correct'].sum()
    total_revision_correct = comparison_stats['revised_correct'].sum()
    total_initial_incorrect = comparison_stats['original_incorrect'].sum()
    total_revision_incorrect = comparison_stats['revised_incorrect'].sum()

    tables = {}
    for table_name, cols in zip(['correctness', 'incorrectness'], [['original_correct', 'revised_correct'], ['original_incorrect', 'revised_incorrect']]):
        data = {'type': ['initial', 'revision']}

        for lang in comparison_stats['lang']:
            data[lang] = [
                comparison_stats.loc[comparison_stats['lang'] == lang, cols[0]].values[0],
                comparison_stats.loc[comparison_stats['lang'] == lang, cols[1]].values[0]
            ]

        data['total'] = [
            sum(comparison_stats[cols[0]]),
            sum(comparison_stats[cols[1]])
        ]

        tables[table_name] = pd.DataFrame(data)

        if args.save_tables:
            save_path = f"{args.results_dir}{table_name}_table.csv"
            tables[table_name].to_csv(save_path, index=False)
            print(f"Saved {table_name.capitalize()} Table to {save_path}")

        if args.display_tables:
            print(f"\n{table_name.capitalize()} Table:")
            print(tables[table_name])

    return tables['correctness'], tables['incorrectness']

if __name__ == "__main__":
    # Main script execution

    df = generate_responses()
    df.to_csv(f"{args.results_dir}initial_result.csv", index=False)
    print("Initial result saved to disk.")

    df_revised = generate_revisions(df)
    df_revised.to_csv(f"{args.results_dir}revised_result.csv", index=False)
    print("Generate revisions result saved to disk.")

    df_final = extract_revisions(df_revised)
    df_final.to_csv(f"{args.results_dir}final_result.csv", index=False)
    print("Extraction result saved to disk.")

    comparison_stats = df_final.groupby('lang')[['is_correct', 'is_correct_rev']].agg(
        original_correct=('is_correct', lambda x: sum(x == True)),
        original_incorrect=('is_correct', lambda x: sum(x == False)),
        revised_correct=('is_correct_rev', lambda x: sum(x == True)),
        revised_incorrect=('is_correct_rev', lambda x: sum(x == False))
    ).reset_index()

    correctness_table, incorrectness_table = transform_comparison_stats(comparison_stats)
    print("Processing completed and saved to disk.")