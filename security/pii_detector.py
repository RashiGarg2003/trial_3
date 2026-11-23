"""
Deterministic PII, confidential-data, and secret detector.

The detector scans user input or generated output for sensitive data
such as:

- Email addresses
- Indian phone numbers
- Aadhaar numbers
- PAN numbers
- Payment-card numbers
- Bank-account indicators
- UPI identifiers
- Passport-like identifiers
- JWT tokens
- API keys
- Password assignments
- Private keys
- Database connection strings

The detector returns a schema-validated PIIDetectionResult.

Important:

- Raw sensitive values are never stored in PIIFinding.
- Detection findings contain only classifications and redaction labels.
- Critical identifiers and secrets cause blocking.
- Lower-risk personal identifiers may be redacted.
- This component does not use an LLM.
"""

import re
from dataclasses import dataclass
from typing import Callable, Optional, Pattern

from security.models import (
    DataClassification,
    PIIAction,
    PIIDetectionResult,
    PIIFinding,
    SecurityDecision,
)


# ---------------------------------------------------------------------
# Internal detector rule
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class DetectionRule:
    """
    Internal deterministic detection rule.

    Attributes:
        entity_type:
            Safe entity category written to PIIFinding.

        pattern:
            Compiled regular expression.

        classification:
            Sensitivity classification.

        action:
            Default treatment.

        replacement:
            Safe replacement.

        reason:
            Safe explanation.

        validator:
            Optional additional validation function.
    """

    entity_type: str
    pattern: Pattern[str]
    classification: DataClassification
    action: PIIAction
    replacement: str
    reason: str
    validator: Optional[
        Callable[[str], bool]
    ] = None


# ---------------------------------------------------------------------
# Additional validators
# ---------------------------------------------------------------------

def passes_luhn_check(
    value: str,
) -> bool:
    """
    Validate a possible payment-card number using the Luhn algorithm.

    Separators are removed before validation.
    """

    digits = re.sub(
        r"\D",
        "",
        value,
    )

    if not (
        13 <= len(digits) <= 19
    ):
        return False

    if len(set(digits)) == 1:
        return False

    total = 0
    reverse_digits = digits[::-1]

    for index, digit_text in enumerate(
        reverse_digits
    ):
        digit = int(digit_text)

        if index % 2 == 1:
            digit *= 2

            if digit > 9:
                digit -= 9

        total += digit

    return total % 10 == 0


def is_valid_aadhaar_candidate(
    value: str,
) -> bool:
    """
    Apply basic structural validation to an Aadhaar candidate.

    This intentionally does not verify the identifier against any
    external authority.

    The detector accepts 12 digits after removing spaces or hyphens and
    rejects obvious repeated-number placeholders.
    """

    digits = re.sub(
        r"\D",
        "",
        value,
    )

    if len(digits) != 12:
        return False

    if len(set(digits)) == 1:
        return False

    # Aadhaar numbers do not normally begin with 0 or 1.
    if digits[0] in {
        "0",
        "1",
    }:
        return False

    return True


def is_valid_indian_phone_candidate(
    value: str,
) -> bool:
    """
    Validate a possible Indian mobile number structurally.
    """

    digits = re.sub(
        r"\D",
        "",
        value,
    )

    if digits.startswith("91") and len(digits) == 12:
        digits = digits[2:]

    if len(digits) != 10:
        return False

    return digits[0] in {
        "6",
        "7",
        "8",
        "9",
    }


def is_valid_bank_account_candidate(
    value: str,
) -> bool:
    """
    Reject obvious placeholders from bank-account-like values.
    """

    digits = re.sub(
        r"\D",
        "",
        value,
    )

    if not (
        9 <= len(digits) <= 18
    ):
        return False

    if len(set(digits)) == 1:
        return False

    return True


# ---------------------------------------------------------------------
# PII detector
# ---------------------------------------------------------------------

class PIIDetector:
    """
    Detect and redact sensitive values.

    Rules are centrally defined and applied consistently to input,
    output, and logging safety checks.
    """

    CLASSIFICATION_PRIORITY = {
        DataClassification.PUBLIC: 0,
        DataClassification.INTERNAL: 1,
        DataClassification.PERSONAL: 2,
        DataClassification.CONFIDENTIAL: 3,
        DataClassification.RESTRICTED: 4,
        DataClassification.SECRET: 5,
    }

    ACTION_PRIORITY = {
        PIIAction.ALLOW: 0,
        PIIAction.REDACT: 1,
        PIIAction.REVIEW: 2,
        PIIAction.BLOCK: 3,
    }

    def __init__(
        self,
        additional_rules: Optional[
            list[DetectionRule]
        ] = None,
    ) -> None:
        """
        Initialize the detector.

        Args:
            additional_rules:
                Optional centrally configured detector rules.

        Application code should not add one-off rules inside agents.
        """

        self.rules = self._build_default_rules()

        if additional_rules:
            self.rules.extend(
                additional_rules
            )

    def _build_default_rules(
        self,
    ) -> list[DetectionRule]:
        """
        Return default deterministic security rules.
        """

        return [
            DetectionRule(
                entity_type="private_key",
                pattern=re.compile(
                    r"-----BEGIN\s+"
                    r"(?:RSA\s+|EC\s+|OPENSSH\s+)?"
                    r"PRIVATE KEY-----",
                    flags=re.IGNORECASE,
                ),
                classification=(
                    DataClassification.SECRET
                ),
                action=PIIAction.BLOCK,
                replacement=(
                    "[PRIVATE_KEY_REDACTED]"
                ),
                reason=(
                    "A private-key marker was detected."
                ),
            ),
            DetectionRule(
                entity_type="jwt_token",
                pattern=re.compile(
                    r"\beyJ[A-Za-z0-9_-]{5,}"
                    r"\.[A-Za-z0-9_-]{5,}"
                    r"\.[A-Za-z0-9_-]{5,}\b"
                ),
                classification=(
                    DataClassification.SECRET
                ),
                action=PIIAction.BLOCK,
                replacement="[JWT_REDACTED]",
                reason=(
                    "A JSON Web Token-like value was detected."
                ),
            ),
            DetectionRule(
                entity_type="database_connection_string",
                pattern=re.compile(
                    r"\b(?:postgres(?:ql)?|mysql|mongodb"
                    r"(?:\+srv)?|mssql|redis)"
                    r"://[^\s]+",
                    flags=re.IGNORECASE,
                ),
                classification=(
                    DataClassification.SECRET
                ),
                action=PIIAction.BLOCK,
                replacement=(
                    "[CONNECTION_STRING_REDACTED]"
                ),
                reason=(
                    "A database connection string was detected."
                ),
            ),
            DetectionRule(
                entity_type="password",
                pattern=re.compile(
                    r"\b(?:password|passwd|pwd)\s*"
                    r"[:=]\s*"
                    r"['\"]?[^\s,'\";]{4,}['\"]?",
                    flags=re.IGNORECASE,
                ),
                classification=(
                    DataClassification.SECRET
                ),
                action=PIIAction.BLOCK,
                replacement="[PASSWORD_REDACTED]",
                reason=(
                    "A password assignment was detected."
                ),
            ),
            DetectionRule(
                entity_type="api_key",
                pattern=re.compile(
                    r"\b(?:api[_-]?key|secret[_-]?key|"
                    r"access[_-]?token|bearer[_-]?token)"
                    r"\s*[:=]\s*"
                    r"['\"]?[A-Za-z0-9_\-./+=]{8,}['\"]?",
                    flags=re.IGNORECASE,
                ),
                classification=(
                    DataClassification.SECRET
                ),
                action=PIIAction.BLOCK,
                replacement="[SECRET_REDACTED]",
                reason=(
                    "An API key or access-token assignment "
                    "was detected."
                ),
            ),
            DetectionRule(
                entity_type="authorization_bearer_token",
                pattern=re.compile(
                    r"\bBearer\s+"
                    r"[A-Za-z0-9_\-./+=]{8,}",
                    flags=re.IGNORECASE,
                ),
                classification=(
                    DataClassification.SECRET
                ),
                action=PIIAction.BLOCK,
                replacement=(
                    "Bearer [TOKEN_REDACTED]"
                ),
                reason=(
                    "An authorization bearer token was detected."
                ),
            ),
            DetectionRule(
                entity_type="email_address",
                pattern=re.compile(
                    r"(?<![\w.+-])"
                    r"[A-Z0-9._%+-]+"
                    r"@[A-Z0-9.-]+"
                    r"\.[A-Z]{2,}"
                    r"(?![A-Z0-9_%+-]|\.[A-Z0-9-])",
                    flags=re.IGNORECASE,
                ),
                classification=(
                    DataClassification.PERSONAL
                ),
                action=PIIAction.REDACT,
                replacement="[EMAIL_REDACTED]",
                reason=(
                    "A personal email address was detected."
                ),
            ),
            DetectionRule(
                entity_type="indian_phone_number",
                pattern=re.compile(
                    r"(?<!\d)"
                    r"(?:\+?91[\s-]?)?"
                    r"[6-9]\d{4}[\s-]?\d{5}"
                    r"(?!\d)"
                ),
                classification=(
                    DataClassification.PERSONAL
                ),
                action=PIIAction.REDACT,
                replacement="[PHONE_REDACTED]",
                reason=(
                    "An Indian mobile-number-like value "
                    "was detected."
                ),
                validator=(
                    is_valid_indian_phone_candidate
                ),
            ),
            DetectionRule(
                entity_type="aadhaar_number",
                pattern=re.compile(
                    r"(?<!\d)"
                    r"[2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4}"
                    r"(?!\d)"
                ),
                classification=(
                    DataClassification.RESTRICTED
                ),
                action=PIIAction.BLOCK,
                replacement="[AADHAAR_REDACTED]",
                reason=(
                    "An Aadhaar-number-like value was detected."
                ),
                validator=(
                    is_valid_aadhaar_candidate
                ),
            ),
            DetectionRule(
                entity_type="pan_number",
                pattern=re.compile(
                    r"(?<![A-Z0-9])"
                    r"[A-Z]{5}[0-9]{4}[A-Z]"
                    r"(?![A-Z0-9])",
                    flags=re.IGNORECASE,
                ),
                classification=(
                    DataClassification.RESTRICTED
                ),
                action=PIIAction.BLOCK,
                replacement="[PAN_REDACTED]",
                reason=(
                    "An Indian PAN-like identifier was detected."
                ),
            ),
            DetectionRule(
                entity_type="payment_card_number",
                pattern=re.compile(
                    r"(?<!\d)"
                    r"(?:\d[ -]?){13,19}"
                    r"(?!\d)"
                ),
                classification=(
                    DataClassification.RESTRICTED
                ),
                action=PIIAction.BLOCK,
                replacement="[PAYMENT_CARD_REDACTED]",
                reason=(
                    "A payment-card number passing structural "
                    "validation was detected."
                ),
                validator=passes_luhn_check,
            ),
            DetectionRule(
                entity_type="upi_identifier",
                pattern=re.compile(
                    r"(?<![\w.-])"
                    r"[A-Z0-9._-]{2,}"
                    r"@"
                    r"(?:upi|ybl|paytm|oksbi|okhdfcbank|"
                    r"okicici|okaxis|ibl|axl|apl)"
                    r"(?![\w.-])",
                    flags=re.IGNORECASE,
                ),
                classification=(
                    DataClassification.CONFIDENTIAL
                ),
                action=PIIAction.BLOCK,
                replacement="[UPI_ID_REDACTED]",
                reason=(
                    "A UPI identifier was detected."
                ),
            ),
            DetectionRule(
                entity_type="ifsc_code",
                pattern=re.compile(
                    r"(?<![A-Z0-9])"
                    r"[A-Z]{4}0[A-Z0-9]{6}"
                    r"(?![A-Z0-9])",
                    flags=re.IGNORECASE,
                ),
                classification=(
                    DataClassification.CONFIDENTIAL
                ),
                action=PIIAction.REVIEW,
                replacement="[IFSC_REDACTED]",
                reason=(
                    "An IFSC-code-like value was detected."
                ),
            ),
            DetectionRule(
                entity_type="bank_account_number",
                pattern=re.compile(
                    r"\b(?:account|a/c|acct)"
                    r"(?:\s+number|\s+no\.?)?"
                    r"\s*[:=-]?\s*"
                    r"(\d[\d -]{7,20}\d)\b",
                    flags=re.IGNORECASE,
                ),
                classification=(
                    DataClassification.RESTRICTED
                ),
                action=PIIAction.BLOCK,
                replacement=(
                    "[BANK_ACCOUNT_REDACTED]"
                ),
                reason=(
                    "A bank-account-number-like value "
                    "was detected."
                ),
                validator=(
                    is_valid_bank_account_candidate
                ),
            ),
            DetectionRule(
                entity_type="indian_passport_number",
                pattern=re.compile(
                    r"(?<![A-Z0-9])"
                    r"[A-Z][1-9][0-9]{6}"
                    r"(?![A-Z0-9])",
                    flags=re.IGNORECASE,
                ),
                classification=(
                    DataClassification.RESTRICTED
                ),
                action=PIIAction.BLOCK,
                replacement="[PASSPORT_REDACTED]",
                reason=(
                    "An Indian-passport-number-like value "
                    "was detected."
                ),
            ),
            DetectionRule(
                entity_type="ipv4_address",
                pattern=re.compile(
                    r"(?<!\d)"
                    r"(?:"
                    r"(?:25[0-5]|2[0-4]\d|1?\d?\d)"
                    r"\.){3}"
                    r"(?:25[0-5]|2[0-4]\d|1?\d?\d)"
                    r"(?!\d)"
                ),
                classification=(
                    DataClassification.INTERNAL
                ),
                action=PIIAction.REVIEW,
                replacement="[IP_ADDRESS_REDACTED]",
                reason=(
                    "An IP address was detected."
                ),
            ),
        ]

    def _create_findings(
        self,
        text: str,
    ) -> list[PIIFinding]:
        """
        Apply all configured rules and return non-overlapping findings.
        """

        candidate_findings: list[
            tuple[int, int, DetectionRule]
        ] = []

        for rule in self.rules:
            for match in rule.pattern.finditer(
                text
            ):
                matched_value = match.group(0)

                value_for_validation = (
                    match.group(1)
                    if match.lastindex
                    else matched_value
                )

                if (
                    rule.validator is not None
                    and not rule.validator(
                        value_for_validation
                    )
                ):
                    continue

                candidate_findings.append(
                    (
                        match.start(),
                        match.end(),
                        rule,
                    )
                )

        # Higher-priority findings win when ranges overlap.
        candidate_findings.sort(
            key=lambda item: (
                item[0],
                -self.ACTION_PRIORITY[
                    item[2].action
                ],
                -(
                    item[1] - item[0]
                ),
            )
        )

        accepted_ranges: list[
            tuple[int, int]
        ] = []

        findings: list[PIIFinding] = []

        for start_index, end_index, rule in (
            candidate_findings
        ):
            overlaps_existing = any(
                start_index < existing_end
                and end_index > existing_start
                for existing_start, existing_end
                in accepted_ranges
            )

            if overlaps_existing:
                continue

            accepted_ranges.append(
                (
                    start_index,
                    end_index,
                )
            )

            findings.append(
                PIIFinding(
                    entity_type=(
                        rule.entity_type
                    ),
                    classification=(
                        rule.classification
                    ),
                    action=rule.action,
                    start_index=start_index,
                    end_index=end_index,
                    redacted_value=(
                        rule.replacement
                    ),
                    confidence=1.0,
                    reason=rule.reason,
                )
            )

        return sorted(
            findings,
            key=lambda finding: (
                finding.start_index
                if finding.start_index
                is not None
                else -1
            ),
        )

    def _redact_text(
        self,
        text: str,
        findings: list[PIIFinding],
    ) -> str:
        """
        Replace findings from right to left so offsets remain valid.
        """

        redacted_text = text

        findings_with_location = [
            finding
            for finding in findings
            if finding.start_index is not None
            and finding.end_index is not None
        ]

        for finding in sorted(
            findings_with_location,
            key=lambda item: (
                item.start_index
                if item.start_index
                is not None
                else -1
            ),
            reverse=True,
        ):
            start_index = finding.start_index
            end_index = finding.end_index

            if (
                start_index is None
                or end_index is None
            ):
                continue

            redacted_text = (
                redacted_text[:start_index]
                + finding.redacted_value
                + redacted_text[end_index:]
            )

        return redacted_text

    def _highest_classification(
        self,
        findings: list[PIIFinding],
    ) -> Optional[DataClassification]:
        """
        Return the highest sensitivity classification.
        """

        if not findings:
            return None

        return max(
            (
                finding.classification
                for finding in findings
            ),
            key=lambda classification: (
                self.CLASSIFICATION_PRIORITY[
                    DataClassification(
                        classification
                    )
                ]
            ),
        )

    def _decision(
        self,
        findings: list[PIIFinding],
    ) -> SecurityDecision:
        """
        Determine the overall privacy decision.
        """

        if not findings:
            return SecurityDecision.ALLOW

        highest_action = max(
            (
                finding.action
                for finding in findings
            ),
            key=lambda action: (
                self.ACTION_PRIORITY[
                    PIIAction(action)
                ]
            ),
        )

        if highest_action == PIIAction.BLOCK:
            return SecurityDecision.BLOCK

        if highest_action == PIIAction.REVIEW:
            return SecurityDecision.REVIEW

        if highest_action == PIIAction.REDACT:
            return (
                SecurityDecision
                .ALLOW_REDACTED
            )

        return SecurityDecision.ALLOW

    def _user_message(
        self,
        decision: SecurityDecision,
        findings: list[PIIFinding],
    ) -> str:
        """
        Create a safe user-facing explanation.
        """

        if decision == SecurityDecision.ALLOW:
            return (
                "No personal, confidential, restricted, "
                "or secret values were detected."
            )

        detected_types = sorted(
            {
                finding.entity_type
                for finding in findings
            }
        )

        safe_types = ", ".join(
            detected_types
        )

        if decision == SecurityDecision.BLOCK:
            return (
                "The request was blocked because it contains "
                "restricted or secret information. Detected "
                f"categories: {safe_types}."
            )

        if decision == SecurityDecision.REVIEW:
            return (
                "The request requires review because it "
                "contains potentially confidential information. "
                f"Detected categories: {safe_types}."
            )

        return (
            "Personal information was redacted before "
            "processing. Detected categories: "
            f"{safe_types}."
        )

    def detect(
        self,
        text: str,
    ) -> PIIDetectionResult:
        """
        Scan text and return a validated privacy decision.

        Args:
            text:
                User question or generated output.

        Returns:
            PIIDetectionResult.

        Raises:
            TypeError:
                If text is not a string.

            ValueError:
                If text is empty.
        """

        if not isinstance(text, str):
            raise TypeError(
                "text must be a string."
            )

        cleaned_text = text.strip()

        if not cleaned_text:
            raise ValueError(
                "text cannot be empty."
            )

        findings = self._create_findings(
            cleaned_text
        )

        redacted_text = self._redact_text(
            text=cleaned_text,
            findings=findings,
        )

        decision = self._decision(
            findings
        )

        return PIIDetectionResult(
            contains_sensitive_data=bool(
                findings
            ),
            highest_classification=(
                self._highest_classification(
                    findings
                )
            ),
            findings=findings,
            redacted_text=redacted_text,
            decision=decision,
            user_message=(
                self._user_message(
                    decision=decision,
                    findings=findings,
                )
            ),
        )


def main() -> None:
    """
    Run a standalone PII detector smoke test.

    Example values are synthetic and used only for testing.
    """

    print("=" * 80)
    print("PII DETECTOR SMOKE TEST")
    print("=" * 80)

    detector = PIIDetector()

    test_cases = [
        (
            "Safe analytical request",
            (
                "Show total revenue by product category."
            ),
        ),
        (
            "Email redaction",
            (
                "Show orders for customer "
                "someone@example.com."
            ),
        ),
        (
            "Restricted identifier blocking",
            (
                "Analyze customer account with PAN "
                "ABCDE1234F."
            ),
        ),
        (
            "Password blocking",
            (
                "Use password=ExampleSecret123 "
                "to connect."
            ),
        ),
    ]

    try:
        for test_name, test_text in test_cases:
            result = detector.detect(
                test_text
            )

            print("\n" + "-" * 80)
            print(test_name)
            print("-" * 80)

            print(
                f"Decision              : "
                f"{result.decision}"
            )

            print(
                f"Sensitive data        : "
                f"{result.contains_sensitive_data}"
            )

            print(
                f"Highest classification: "
                f"{result.highest_classification}"
            )

            print(
                f"Redacted text         : "
                f"{result.redacted_text}"
            )

            detected_categories = [
                finding.entity_type
                for finding in result.findings
            ]

            print(
                f"Detected categories   : "
                f"{detected_categories}"
            )

            print(
                f"User message          : "
                f"{result.user_message}"
            )

        print("\n" + "=" * 80)
        print("RESULT: PII DETECTOR IS WORKING")
        print("=" * 80)

    except Exception as exc:
        print("\n" + "=" * 80)
        print("RESULT: PII DETECTOR FAILED")
        print("=" * 80)

        print(
            f"{type(exc).__name__}: {exc}"
        )

        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()