DEFAULT_SQL_PATH = "sqlite:///ordercost.sqlite"
DEFAULT_ORDER_TABLE_DESCRP = (
    "This table gives information regarding the order master data\n"
    "Each order has a associated line of business and cost category\n"
    "The orders are also separate for budget forecast and actual"
)
DEFAULT_COST_TABLE_DESCRP = (
    "This table gives information regarding the cost by order for the month\n"
    "Cost is broken out into budget, forecast, and actual\n"
    "Additional data about the orders can be found in the order_table\n"
)
DEFAULT_LC_TOOL_DESCRP = "Useful for when you want to answer queries about order costs."
DEFAULT_INGEST_DOCUMENT = (
    "The HR Line of Business had a budget of 25000 for June, their forecast was 22000 and actual 24000. \n"
    "The YTD Expense spend is 500000 and Capital is 750000"
)
