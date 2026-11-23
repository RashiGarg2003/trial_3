"""
Shared Pydantic schemas for the Multi-Agent Data Analytics System.

These models define the structured contracts exchanged between:

1. Intent Agent
2. Planning Agent
3. SQL Generation Agent
4. Analysis Agent
5. Reporting Agent

The models do not contain hardcoded database tables, columns,
metrics, filters, or business rules.
"""

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


AnalysisType = Literal[
    "aggregation",
    "comparison",
    "trend_analysis",
    "ranking",
    "root_cause_analysis",
    "anomaly_detection",
    "distribution_analysis",
    "general_analysis",
]

SortOrder = Literal[
    "ascending",
    "descending",
]

StepType = Literal[
    "schema_inspection",
    "data_retrieval",
    "aggregation",
    "comparison",
    "trend_analysis",
    "ranking",
    "root_cause_analysis",
    "anomaly_detection",
    "statistical_analysis",
    "visualization",
    "reporting",
    "general_analysis",
]

StepStatus = Literal[
    "pending",
    "running",
    "completed",
    "failed",
    "skipped",
]

ChartType = Literal[
    "line",
    "bar",
    "horizontal_bar",
    "area",
    "scatter",
    "pie",
    "table",
    "none",
]


class StrictModel(BaseModel):
    """
    Base model used by all shared schemas.

    Unknown fields are rejected so that agents cannot silently add
    unsupported output fields.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class TimePeriod(StrictModel):
    """
    Time-related information extracted from a user question.
    """

    start_date: Optional[str] = Field(
        default=None,
        description=(
            "Start date or start period explicitly identified "
            "from the user's request."
        ),
    )

    end_date: Optional[str] = Field(
        default=None,
        description=(
            "End date or end period explicitly identified "
            "from the user's request."
        ),
    )

    current_period: Optional[str] = Field(
        default=None,
        description=(
            "Primary period requested by the user, when expressed "
            "as a business period rather than an exact date."
        ),
    )

    comparison_period: Optional[str] = Field(
        default=None,
        description=(
            "Previous or comparison period requested by the user."
        ),
    )

    granularity: Optional[str] = Field(
        default=None,
        description=(
            "Requested time granularity, such as day, week, month, "
            "quarter, or year."
        ),
    )


class FilterCondition(StrictModel):
    """
    One structured filter extracted from the user request.

    A filter is represented generically so it can work with different
    databases and business domains.
    """

    field: str = Field(
        min_length=1,
        description=(
            "Business field or concept to which the filter applies."
        ),
    )

    operator: str = Field(
        min_length=1,
        description=(
            "Filter operation, such as equals, contains, greater_than, "
            "between, or in."
        ),
    )

    value: Any = Field(
        description="Value used by the filter."
    )


class IntentResult(StrictModel):
    """
    Structured output of the Intent Understanding Agent.
    """

    original_question: str = Field(
        min_length=1,
        description="Original question entered by the user.",
    )

    rewritten_question: str = Field(
        min_length=1,
        description=(
            "Clear analytical version of the original question."
        ),
    )

    analysis_type: AnalysisType = Field(
        description="Primary analysis requested by the user."
    )

    metrics: list[str] = Field(
        default_factory=list,
        description=(
            "Business measures requested by the user."
        ),
    )

    dimensions: list[str] = Field(
        default_factory=list,
        description=(
            "Grouping, segmentation, or comparison dimensions."
        ),
    )

    filters: list[FilterCondition] = Field(
        default_factory=list,
        description=(
            "Explicit filtering conditions extracted from the request."
        ),
    )

    time_period: TimePeriod = Field(
        default_factory=TimePeriod,
        description="Extracted time requirements.",
    )

    sort_order: Optional[SortOrder] = Field(
    default=None,
    description=(
        "Sorting direction required to satisfy the analytical "
        "objective. Leading or highest-value rankings use "
        "descending order. Trailing or lowest-value rankings use "
        "ascending order. Use null when ordering is not relevant."
    ),
)

    limit: Optional[int] = Field(
        default=None,
        ge=1,
        le=10000,
        description=(
            "Maximum number of requested results."
        ),
    )

    clarification_required: bool = Field(
        default=False,
        description=(
            "Whether essential information is missing."
        ),
    )

    clarification_question: Optional[str] = Field(
        default=None,
        description=(
            "Question to ask when clarification is required."
        ),
    )

    @model_validator(mode="after")
    def validate_clarification(
        self,
    ) -> "IntentResult":
        """
        Ensure clarification fields are internally consistent.
        """

        if (
            self.clarification_required
            and not self.clarification_question
        ):
            raise ValueError(
                "clarification_question is required when "
                "clarification_required is True."
            )

        if (
            not self.clarification_required
            and self.clarification_question is not None
        ):
            raise ValueError(
                "clarification_question must be None when "
                "clarification_required is False."
            )

        return self


class PlanStep(StrictModel):
    """
    One atomic step created by the Planning Agent.
    """

    step_id: int = Field(
        ge=1,
        description="Unique positive step number.",
    )

    title: str = Field(
        min_length=1,
        description="Short name of the analytical step.",
    )

    description: str = Field(
        min_length=1,
        description=(
            "Detailed explanation of what the step must accomplish."
        ),
    )

    step_type: StepType = Field(
        description="Type of analytical operation."
    )

    depends_on: list[int] = Field(
        default_factory=list,
        description=(
            "Step IDs that must complete before this step."
        ),
    )

    required_metrics: list[str] = Field(
        default_factory=list,
        description="Metrics required by this step.",
    )

    required_dimensions: list[str] = Field(
        default_factory=list,
        description="Dimensions required by this step.",
    )

    required_data: list[str] = Field(
        default_factory=list,
        description=(
            "Conceptual data requirements. These are not assumed "
            "to be physical table or column names."
        ),
    )

    expected_output: str = Field(
        min_length=1,
        description=(
            "Description of the expected result from this step."
        ),
    )

    requires_sql: bool = Field(
        default=False,
        description=(
            "Whether this step requires database retrieval."
        ),
    )

    status: StepStatus = Field(
        default="pending",
        description="Current execution status.",
    )


class AnalysisPlan(StrictModel):
    """
    Structured output of the Planning Agent.
    """

    objective: str = Field(
        min_length=1,
        description="Overall analytical goal.",
    )

    strategy: str = Field(
        min_length=1,
        description=(
            "Short explanation of how the system will answer "
            "the business question."
        ),
    )

    steps: list[PlanStep] = Field(
        min_length=1,
        description="Ordered analytical execution steps.",
    )

    final_output_requirements: list[str] = Field(
        default_factory=list,
        description=(
            "Information expected in the final response."
        ),
    )

    @model_validator(mode="after")
    def validate_plan_steps(
        self,
    ) -> "AnalysisPlan":
        """
        Validate step IDs and dependencies without applying business rules.
        """

        step_ids = [
            step.step_id
            for step in self.steps
        ]

        if len(step_ids) != len(set(step_ids)):
            raise ValueError(
                "Every plan step must have a unique step_id."
            )

        available_step_ids = set(step_ids)

        for step in self.steps:
            if step.step_id in step.depends_on:
                raise ValueError(
                    f"Step {step.step_id} cannot depend on itself."
                )

            unknown_dependencies = (
                set(step.depends_on)
                - available_step_ids
            )

            if unknown_dependencies:
                raise ValueError(
                    f"Step {step.step_id} contains unknown "
                    f"dependencies: "
                    f"{sorted(unknown_dependencies)}"
                )

            future_dependencies = [
                dependency
                for dependency in step.depends_on
                if dependency >= step.step_id
            ]

            if future_dependencies:
                raise ValueError(
                    f"Step {step.step_id} cannot depend on "
                    "itself or a later step."
                )

        return self


class SQLGenerationRequest(StrictModel):
    """
    Input supplied to the SQL Generation Agent.
    """

    user_question: str = Field(
        min_length=1,
        description="Original business question.",
    )

    intent: IntentResult = Field(
        description="Validated output from the Intent Agent."
    )

    plan: AnalysisPlan = Field(
        description="Validated output from the Planning Agent."
    )

    database_type: str = Field(
        min_length=1,
        description="Target SQL dialect or database engine.",
    )

    database_schema: str = Field(
        min_length=1,
        description=(
            "Actual database schema extracted by SchemaManager."
        ),
    )

    previous_sql: Optional[str] = Field(
        default=None,
        description=(
            "Previous SQL query when the agent is correcting an error."
        ),
    )

    previous_error: Optional[str] = Field(
        default=None,
        description=(
            "Database or validation error from a previous SQL attempt."
        ),
    )


class SQLGenerationResult(StrictModel):
    """
    Structured output of the SQL Generation Agent.
    """

    sql: str = Field(
        min_length=1,
        description="Generated read-only SQL query.",
    )

    explanation: str = Field(
        min_length=1,
        description=(
            "Short explanation of the query logic."
        ),
    )

    referenced_tables: list[str] = Field(
        default_factory=list,
        description=(
            "Database tables or views used by the query."
        ),
    )

    expected_columns: list[str] = Field(
        default_factory=list,
        description=(
            "Columns expected in the query result."
        ),
    )

    assumptions: list[str] = Field(
        default_factory=list,
        description=(
            "Explicit assumptions made during SQL generation."
        ),
    )

    requires_clarification: bool = Field(
        default=False,
        description=(
            "Whether reliable SQL cannot be generated without "
            "additional information."
        ),
    )

    clarification_question: Optional[str] = Field(
        default=None,
        description=(
            "Question to ask if SQL generation requires clarification."
        ),
    )

    @model_validator(mode="after")
    def validate_sql_clarification(
        self,
    ) -> "SQLGenerationResult":
        """
        Ensure SQL clarification fields are consistent.
        """

        if (
            self.requires_clarification
            and not self.clarification_question
        ):
            raise ValueError(
                "clarification_question is required when "
                "requires_clarification is True."
            )

        if (
            not self.requires_clarification
            and self.clarification_question is not None
        ):
            raise ValueError(
                "clarification_question must be None when "
                "requires_clarification is False."
            )

        return self


class QueryExecutionResult(StrictModel):
    """
    Serializable result returned after SQL execution.
    """

    columns: list[str] = Field(
        default_factory=list,
        description="Names of returned columns.",
    )

    rows: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Rows returned by the database.",
    )

    row_count: int = Field(
        ge=0,
        description="Number of rows included in the result.",
    )

    execution_time_ms: float = Field(
        ge=0,
        description="Database execution time in milliseconds.",
    )

    truncated: bool = Field(
        default=False,
        description=(
            "Whether the result exceeded the configured row limit."
        ),
    )

    query: str = Field(
        min_length=1,
        description="SQL query executed by the database.",
    )


class AnalysisFinding(StrictModel):
    """
    One evidence-based finding produced by the Analysis Agent.
    """

    title: str = Field(
        min_length=1,
        description="Short finding title.",
    )

    description: str = Field(
        min_length=1,
        description="Explanation of the finding.",
    )

    evidence: list[str] = Field(
        default_factory=list,
        description=(
            "Values or observations supporting the finding."
        ),
    )

    importance: Literal[
        "low",
        "medium",
        "high",
    ] = Field(
        default="medium",
        description="Business importance of the finding.",
    )


class AnalysisResult(StrictModel):
    """
    Structured output of the Analysis Agent.
    """

    summary: str = Field(
        min_length=1,
        description="Overall interpretation of the query results.",
    )

    findings: list[AnalysisFinding] = Field(
        default_factory=list,
        description="Evidence-based analytical findings.",
    )

    trends: list[str] = Field(
        default_factory=list,
        description="Detected trends.",
    )

    anomalies: list[str] = Field(
        default_factory=list,
        description="Detected unusual observations.",
    )

    possible_drivers: list[str] = Field(
        default_factory=list,
        description=(
            "Possible contributors supported by available data."
        ),
    )

    limitations: list[str] = Field(
        default_factory=list,
        description=(
            "Limitations of the analysis and available evidence."
        ),
    )


class ChartSpecification(StrictModel):
    """
    Generic chart configuration returned for the interface.
    """

    chart_type: ChartType = Field(
        default="none",
        description="Recommended chart type.",
    )

    title: str = Field(
        min_length=1,
        description="Chart title.",
    )

    x_column: Optional[str] = Field(
        default=None,
        description="Column used for the horizontal axis.",
    )

    y_columns: list[str] = Field(
        default_factory=list,
        description=(
            "One or more columns used for the vertical axis."
        ),
    )

    category_column: Optional[str] = Field(
        default=None,
        description=(
            "Optional column used for color or series grouping."
        ),
    )

    rationale: str = Field(
        min_length=1,
        description=(
            "Reason this chart is appropriate for the result."
        ),
    )


class Recommendation(StrictModel):
    """
    One recommendation produced by the Reporting Agent.
    """

    action: str = Field(
        min_length=1,
        description="Recommended action.",
    )

    rationale: str = Field(
        min_length=1,
        description=(
            "Evidence-based reason for the recommendation."
        ),
    )

    priority: Literal[
        "low",
        "medium",
        "high",
    ] = Field(
        default="medium",
        description="Recommended action priority.",
    )


class FinalReport(StrictModel):
    """
    Structured output of the Reporting Agent.
    """

    business_summary: str = Field(
        min_length=1,
        description="Business-friendly analytical summary.",
    )

    key_findings: list[str] = Field(
        default_factory=list,
        description="Most important analytical findings.",
    )

    root_cause_indicators: list[str] = Field(
        default_factory=list,
        description=(
            "Data-supported root-cause indicators. These are not "
            "presented as proven causes unless evidence is sufficient."
        ),
    )

    recommendations: list[Recommendation] = Field(
        default_factory=list,
        description="Evidence-based recommendations.",
    )

    chart: Optional[ChartSpecification] = Field(
        default=None,
        description="Optional visualization specification.",
    )

    limitations: list[str] = Field(
        default_factory=list,
        description="Relevant limitations and assumptions.",
    )