"""测试场景模块。"""

# 场景 1：企业评估阶段
SCENARIO_1_QUERY = """
We are evaluating your platform for our enterprise. We need to know the specific
API rate limits for the Enterprise plan, your data export options for a 2GB migration,
the security compliance standards you meet, and if you support ACH payments.
"""

# 场景 2：实施规划阶段
SCENARIO_2_QUERY = """
We're moving forward with implementation planning. I need to compare API rate limits
between Pro and Enterprise plans to decide on our tier, confirm the Salesforce
integration capabilities we discussed, understand what data export options you provide
for our migration needs, and verify the payment methods including ACH since our
accounting team prefers that for monthly billing.
"""

# 场景 3：最终确认阶段
SCENARIO_3_QUERY = """
Before finalizing our Pro plan purchase, I need complete validation on: your security
compliance framework including SOC2 requirements, the exact API rate limits for the
Pro plan we're purchasing, confirmation of the Salesforce integration features, all
supported payment methods since we want to use ACH transfers, and your data export
capabilities for our future migration planning.
"""

SCENARIO_RUNS = [
    {
        "title": "Scenario 1: Enterprise Platform Evaluation",
        "query": SCENARIO_1_QUERY,
    },
    {
        "title": "Scenario 2: Implementation Planning",
        "query": SCENARIO_2_QUERY,
    },
    {
        "title": "Scenario 3: Pre-Purchase Comprehensive Review",
        "query": SCENARIO_3_QUERY,
    },
]