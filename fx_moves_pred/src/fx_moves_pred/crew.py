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
            allow_delegation=True,
            max_iter=6,
            system_message="You MUST use your search tools for every task. When using the Serper search tool, ALWAYS pass only a plain string as the search query. DO NOT wrap it in a dictionary."
        )

    # @agent
    # def corporate_credit_specialist(self) -> Agent:
    #     return Agent(
    #         config=self.agents_config['corporate_credit_specialist'],
    #         #tools=[self.serper_tool, self.website_tool],
    #         verbose=True,
    #         allow_delegation=False,
    #         max_iter=2
    #         #system_message="You MUST use your search tools for every task. Never provide general knowledge.",
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

    @task
    def research_fi_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_fi_task'],
            agent=self.fixed_income_specalist(),
            #tools=[self.serper_tool, self.website_tool],
            output_file='research_fi_task_full.md'
        )
    
    # @task
    # def research_cc_task(self) -> Task:
    #     return Task(
    #         config=self.tasks_config['research_cc_task'],
    #         agent=self.corporate_credit_specialist(),
    #         async_execution=False
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


# from crewai import Agent, Crew, Process, Task
# from crewai.project import CrewBase, agent, crew, task
# from crewai.agents.agent_builder.base_agent import BaseAgent
# from crewai_tools import SerperDevTool, WebsiteSearchTool
# from typing import List
# # If you want to run a snippet of code before or after the crew starts,
# # you can use the @before_kickoff and @after_kickoff decorators
# # https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

# @CrewBase
# class FxMovesPred():
#     """FxMovesPred crew"""

#     agents: List[BaseAgent]
#     tasks: List[Task]

#     serper_tool = SerperDevTool()
#     website_tool = WebsiteSearchTool()

#     # Learn more about YAML configuration files here:
#     # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
#     # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
#     # If you would like to add tools to your agents, you can learn more about it here:
#     # https://docs.crewai.com/concepts/agents#agent-tools
#     @agent
#     def fixed_income_specalist(self) -> Agent:
#         return Agent(
#             config=self.agents_config['fixed_income_specalist'], # type: ignore[index]
#             tools=[self.serper_tool, self.website_tool],
#             verbose=True
#         )

#     @agent
#     def corporate_credit_specialist(self) -> Agent:
#         return Agent(
#             config=self.agents_config['corporate_credit_specialist'], # type: ignore[index]
#             tools=[self.serper_tool, self.website_tool],
#             verbose=True
#         )
    
#     @agent
#     def fx_strategist(self) -> Agent:
#         return Agent(
#             config=self.agents_config['fx_strategist'], # type: ignore[index]
#             tools=[self.serper_tool, self.website_tool],
#             verbose=True
#         )
    
#     @agent
#     def global_economist(self) -> Agent:
#         return Agent(
#             config=self.agents_config['global_economist'], # type: ignore[index]
#             tools=[self.serper_tool, self.website_tool],
#             verbose=True
#         )
    
#     @agent
#     def portfolio_manager(self) -> Agent:
#         return Agent(
#             config=self.agents_config['portfolio_manager'], # type: ignore[index]
#             tools=[self.serper_tool, self.website_tool],
#             verbose=True
#         )

#     # To learn more about structured task outputs,
#     # task dependencies, and task callbacks, check out the documentation:
#     # https://docs.crewai.com/concepts/tasks#overview-of-a-task
#     @task
#     def research_fi_task(self) -> Task:
#         return Task(
#             config=self.tasks_config['research_fi_task'], # type: ignore[index]
#             agent=self.fixed_income_specalist()
#         )
    
#     @task
#     def research_cc_task(self) -> Task:
#         return Task(
#             config=self.tasks_config['research_cc_task'], # type: ignore[index]
#             agent=self.corporate_credit_specialist()
#         )
    
#     @task
#     def research_fx_task(self) -> Task:
#         return Task(
#             config=self.tasks_config['research_fx_task'], # type: ignore[index]
#             agent=self.fx_strategist()
#         )

#     @task
#     def analyze_related_markets_task(self) -> Task:
#         return Task(
#             config=self.tasks_config['analyze_related_markets_task'], # type: ignore[index]
#             agent=self.fx_strategist()
#         )
    
#     @task
#     def verify_macro_logic(self) -> Task:
#         return Task(
#             config=self.tasks_config['verify_macro_logic'], # type: ignore[index]
#             agent=self.global_economist()
#         )
    
#     @task
#     def final_review(self) -> Task:
#         return Task(
#             config=self.tasks_config['final_review'], # type: ignore[index]
#             agent=self.portfolio_manager(),
#             output_file='report.md'
#         )

#     @crew
#     def crew(self) -> Crew:
#         """Creates the FxMovesPred crew"""
#         # To learn how to add knowledge sources to your crew, check out the documentation:
#         # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

#         return Crew(
#             agents=self.agents, # Automatically created by the @agent decorator
#             tasks=self.tasks, # Automatically created by the @task decorator
#             process=Process.sequential,
#             verbose=True,
#             autonomous=True
#             # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
#         )
