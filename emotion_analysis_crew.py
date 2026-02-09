# emotion_analysis_crew.py
"""
Emotion Analysis Crew (Advanced Data Pipeline)
----------------------------------------------
A comprehensive multi-agent system that processes the Hugging Face 'emotion' dataset.
It simulates a full data science team:
1. Data Preprocessor: Cleans and prepares data.
2. Emotion Classifier: Uses LLM to classify text similar/different to ground truth.
3. Emotion Analyst: Analyzes statistical distributions.
4. Insight Reporter: Generates actionable advice based on findings.

Requirement:
    - Ollama running locally.
    - `pip install crewai pandas datasets`
"""

import pandas as pd
from datasets import load_dataset
from crewai import Agent, Task, Crew
from crewai.llm import LLM


def main():
    # 1. Validation & Data Loading
    print("Loading 'emotion' dataset from Hugging Face...")
    dataset = load_dataset("emotion", split="train")
    df = pd.DataFrame(dataset)

    # Helper: Pre-calculate basic stats to feed into the Analyst
    emotion_counts = df["label"].value_counts()
    label_names = dataset.features["label"].names
    summary = "\n".join(
        [
            f"{label_names[i]}: {emotion_counts.get(i, 0)}"
            for i in range(len(label_names))
        ]
    )

    # 2. LLM Configuration
    llm = LLM(model="ollama/qwen2.5:0.5b-instruct", base_url="http://localhost:11434")

    # 3. Agent Definitions
    preprocessor = Agent(
        role="Data Preprocessor",
        goal="Clean and prepare the emotion dataset for analysis",
        backstory="You are an expert in data cleaning and preprocessing.",
        llm=llm,
        verbose=True,
    )

    classifier = Agent(
        role="Emotion Classifier",
        goal="Classify the emotions in sample texts using the LLM",
        backstory="You are skilled at using LLMs for text classification.",
        llm=llm,
        verbose=True,
    )

    analyzer = Agent(
        role="Emotion Analyst",
        goal="Analyze the statistics and distribution and patterns in the emotion dataset",
        backstory="You are a data scientist specializing in emotion analysis",
        llm=llm,
        verbose=True,
    )

    reporter = Agent(
        role="Insight Reporter",
        goal="Generate actionable insights from the emotion dataset patterns",
        backstory="You are an expert at summarizing data findings",
        llm=llm,
        verbose=True,
    )

    # 4. Task Definitions
    # Task A: Preprocessing
    preprocess_task = Task(
        description="Clean and preprocess the emotion dataset. Remove empty texts and normalize text.",
        agent=preprocessor,
        expected_output="A cleaned DataFrame with normalized text.",
    )

    # Task B: Classification (Sample)
    sample_texts = df["text"].sample(5, random_state=42).tolist()
    classify_task = Task(
        description=f"Classify the emotions in these sample texts: {sample_texts}",
        agent=classifier,
        expected_output="A list of predicted emotions for the sample texts.",
        context=[preprocess_task],
    )

    # Task C: Analysis
    analysis_task = Task(
        description="Review the emotion dataset and summarize the distribution of emotions. Stats: "
        + summary,
        agent=analyzer,
        expected_output="A summary of emotion counts and notable patterns",
        context=[classify_task],
    )

    # Task D: Reporting
    insight_task = Task(
        description="Based on the analysis, provide 3 actionable insights for improving emotional well-being.",
        agent=reporter,
        expected_output="3 actionable insights",
        context=[analysis_task],
    )

    # 5. Pipeline Execution
    crew = Crew(
        agents=[preprocessor, classifier, analyzer, reporter],
        tasks=[preprocess_task, classify_task, analysis_task, insight_task],
        verbose=True,
    )

    print("\n🚀 Starting emotion dataset analysis...\n")
    result = crew.kickoff()
    print("\n" + "=" * 50)
    print("FINAL RESULT:")
    print("=" * 50)
    print(result)
    print("=" * 50)


if __name__ == "__main__":
    main()
