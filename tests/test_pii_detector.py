from security.pii_detector import PIIDetector
from security.models import SecurityDecision


def test_email_followed_by_sentence_punctuation_is_redacted():
    detector = PIIDetector()

    result = detector.detect(
        "Show orders for customer person@example.com."
    )

    assert result.decision == (
        SecurityDecision.ALLOW_REDACTED
    )
    assert (
        result.redacted_text
        == "Show orders for customer [EMAIL_REDACTED]."
    )
