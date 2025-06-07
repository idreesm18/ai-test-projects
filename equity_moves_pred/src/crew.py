from typing import List
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent

@CrewBase
class TradingTeam():
    """TradingTeam crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def equity_analyst(self) -> Agent:
        return Agent(
            role="Equity Analyst",
            goal="Analyze specific stocks and their performance",
            backstory="You specialize in fundamental and technical analysis of individual equities.",
            verbose=True
        )

    @agent
    def macro_analyst(self) -> Agent:
        return Agent(
            role="Macro Analyst",
            goal="Assess macroeconomic trends that could impact market performance",
            backstory="You track inflation, interest rates, employment data, and global events affecting markets.",
            verbose=True
        )

    @agent
    def quant_analyst(self) -> Agent:
        return Agent(
            role="Quantitative Analyst",
            goal="Use quantitative methods to detect market inefficiencies or patterns",
            backstory="You apply statistical models and machine learning to market data for insights.",
            verbose=True
        )

    @task
    def equity_task(self) -> Task:
        return Task(
            description="Evaluate the current valuation and recent price movement of a {stock}.",
            expected_output="A concise report on stock performance, valuation metrics, and sentiment.",
            agent=self.equity_analyst(),
        )

    @task
    def macro_task(self) -> Task:
        return Task(
            description="Summarize recent macroeconomic indicators and interpret how they affect {stock}.",
            expected_output="A macroeconomic snapshot with key takeaways relevant to investment strategies.",
            agent=self.macro_analyst(),
        )

    @task
    def quant_task(self) -> Task:
        return Task(
            description="Detect any unusual price-volume behavior or statistical anomalies in {stock}'s data.",
            expected_output="A brief on any notable market patterns or statistical signals.",
            agent=self.quant_analyst(),
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.equity_analyst(), self.macro_analyst(), self.quant_analyst()],
            tasks=[self.equity_task(), self.macro_task(), self.quant_task()],
            process=Process.sequential,
            verbose=True,
            autonomous=True
        )
