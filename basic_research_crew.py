# basic_research_crew.py
"""
Basic Research Crew (Ollama + CrewAI)
-------------------------------------
This script demonstrates a minimal "Hello World" example of a multi-agent system using CrewAI
and a local Large Language Model (LLM) via Ollama.

Agents:
    1. Researcher: Searches for information on a given topic (simulated via LLM knowledge).
    2. Writer: Summarizes the researcher's findings into a concise paragraph.

Workflow:
    [User Input] -> [Researcher Agent] -> [Writer Agent] -> [Final Summary]

Usage:
    python basic_research_crew.py "future of AI"
"""

import sys
from crewai import Agent, Task, Crew
from crewai.llm import LLM


def main():
    """
    Main execution function.
    1. Sets up the local LLM connection (Ollama).
    2. Defines Agents and their specific roles/goals.
    3. Defines Tasks and links them to agents.
    4. Executes the crew and prints the result.
    """

    # 1. LLM Setup
    # Ensure Ollama is running locally: `ollama run qwen2.5:0.5b-instruct`
    llm = LLM(model="ollama/qwen2.5:0.5b-instruct", base_url="http://localhost:11434")

    # Get topic from command line, default to "benefits of reading"
    topic = "benefits of reading"
    if len(sys.argv) > 1:
        topic = " ".join(sys.argv[1:])

    # 2. Agent Definition
    researcher = Agent(
        role="Researcher",
        goal=f"Research and summarize key points about {topic}",
        backstory="You are a skilled researcher who finds accurate information",
        llm=llm,
        verbose=True,
    )

    writer = Agent(
        role="Writer",
        goal=f"Write a clear summary about {topic}",
        backstory="You are a clear, concise writer",
        llm=llm,
        verbose=True,
    )

    # 3. Task Definition
    research_task = Task(
        description=f"Research {topic} and provide 5 key bullet points",
        agent=researcher,
        expected_output="5 bullet points about the topic",
    )

    write_task = Task(
        description=f"Write a 100-word paragraph about {topic} using the research",
        agent=writer,
        expected_output="A clear 100-word paragraph",
        context=[research_task],  # Waits for research_task to finish
    )

    # 4. Crew Execution
    crew = Crew(
        agents=[researcher, writer], tasks=[research_task, write_task], verbose=True
    )

    # Execute
    print(f"\n🚀 Starting research and writing about: {topic}\n")
    result = crew.kickoff()

    print("\n" + "=" * 50)
    print("FINAL RESULT:")
    print("=" * 50)
    print(result)
    print("=" * 50)


if __name__ == "__main__":
    main()
