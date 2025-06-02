from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai_tools import SerperDevTool, WebsiteSearchTool
from typing import List

@CrewBase
class FxMovesPred():
    """FxMovesPred crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    serper_tool = SerperDevTool()
    #website_tool = WebsiteSearchTool()

    @agent
    def fixed_income_specalist(self) -> Agent:
        return Agent(
            config=self.agents_config['fixed_income_specalist'],
            tools=[self.serper_tool],
            verbose=True,
            max_iter=3,
            system_message="You MUST use your Serper tool for every task."
        )

    # @agent
    # def corporate_credit_specialist(self) -> Agent:
    #     return Agent(
    #         config=self.agents_config['corporate_credit_specialist'],
    #         tools=[self.serper_tool],
    #         verbose=True,
    #         max_iter=3,
    #         system_message="You MUST use your Serper tool for every task."
    #     )
    
    # @agent
    # def fx_strategist(self) -> Agent:
    #     return Agent(
    #         config=self.agents_config['fx_strategist'],
    #         #tools=[self.serper_tool, self.website_tool],
    #         verbose=True,
    #         allow_delegation=True,
    #         system_message="You may ask a coworker a specific question, but do not conclude until you’ve received and included their answer in your response.",
    #         max_iter=2
    #         #system_message="You MUST use your search tools first, then create specific trades with exact numbers.",
    #     )
    
    # @agent
    # def global_economist(self) -> Agent:
    #     return Agent(
    #         config=self.agents_config['global_economist'],
    #         #tools=[self.serper_tool, self.website_tool],
    #         verbose=True,
    #         allow_delegation=True,
    #         system_message="You may ask a coworker a specific question, but do not conclude until you’ve received and included their answer in your response.",
    #         max_iter=2
    #     )
    
    # @agent
    # def portfolio_manager(self) -> Agent:
    #     return Agent(
    #         config=self.agents_config['portfolio_manager'],
    #         #tools=[self.serper_tool, self.website_tool],
    #         verbose=True,
    #         allow_delegation=True,
    #         system_message="You may ask a coworker a specific question, but do not conclude until you’ve received and included their answer in your response.",
    #         max_iter=2
    #     )

    # @task
    # def research_fi_task(self, input_text: str) -> Task:
    #     return Task(
    #         config={**self.tasks_config['research_fi_task'], 'input': input_text},
    #         agent=self.fixed_income_specalist(),
    #         output_file='research_fi_task_country.md'
    #     )
    
    def create_research_fi_task(self, serper_output):
        return Task(
            description=self.tasks_config['research_fi_task']["description"],
            expected_output=self.tasks_config['research_fi_task']["expected_output"],
            agent=self.fixed_income_specalist(),
            input=f"Use this data:\n{serper_output}",
            output_file='research_fi_task_country.md'
        )
    
    # @task
    # def research_cc_task(self) -> Task:
    #     return Task(
    #         config=self.tasks_config['research_cc_task'],
    #         agent=self.corporate_credit_specialist(),
    #         output_file='research_cc_task_country.md'
    #     )
    
    # @task
    # def research_fx_task(self) -> Task:
    #     return Task(
    #         config=self.tasks_config['research_fx_task'],
    #         agent=self.fx_strategist(),
    #         async_execution=False
    #     )

    # @task
    # def analyze_related_markets_task(self) -> Task:
    #     return Task(
    #         config=self.tasks_config['analyze_related_markets_task'],
    #         agent=self.fx_strategist(),
    #         async_execution=False
    #     )
    
    # @task
    # def verify_macro_logic(self) -> Task:
    #     return Task(
    #         config=self.tasks_config['verify_macro_logic'],
    #         agent=self.global_economist(),
    #         async_execution=False
    #     )
    
    # @task
    # def final_review(self) -> Task:
    #     return Task(
    #         config=self.tasks_config['final_review'],
    #         agent=self.portfolio_manager(),
    #         output_file='report.md',
    #         async_execution=False
    #     )

    def run_pipeline(self, inputs):
        query = f"Recent news and trends in {inputs['country']}'s fixed income market before {inputs['current_time']}"
        serper_output = self.serper_tool.run(query=query)
        
        research_fi_task = self.create_research_fi_task(serper_output)
        fixed_income_agent = self.fixed_income_specalist()
        print("DEBUG: Task input text:", research_fi_task)
        research_output = fixed_income_agent.execute_task(research_fi_task)

        with open("pipeline_output.md", "w", encoding="utf-8") as f:
            f.write(f"# Fixed Income Research Output ({inputs['country']})\n\n")
            f.write("## Research Task Output:\n")
            f.write(research_output + "\n\n")
        
        return None

    @crew
    def crew(self) -> Crew:
        """Creates the FxMovesPred crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            autonomous=True
        )