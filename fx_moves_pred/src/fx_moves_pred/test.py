from crewai_tools import SerperDevTool

def test_serper_tool():
    tool = SerperDevTool()
    query = "latest developments in US 10Y treasury yields January 2022"
    result = tool.run()  # pass query as a keyword argument
    print("Tool output:\n", result)

if __name__ == "__main__":
    test_serper_tool()
