from monitoring.logger import RequestLogger
from monitoring.metrics import MonitoringMetrics


def main():

    print("=" * 70)

    print("TESTING MONITORING SYSTEM")

    print("=" * 70)

    logger = RequestLogger()

    # Start request
    request = logger.start_request(
        "What is an A1C test?"
    )

    # Simulated retrieved documents
    documents = [

        {
            "title": "A1C",

            "category": "Diagnostic Tests",

            "language": "en",

            "source": "https://medlineplus.gov/a1c.html",

            "content": (
                "A1C is a blood test that measures "
                "average blood glucose levels."
            )
        },

        {
            "title": "Diabetes Type 2",

            "category": "Endocrine System",

            "language": "en",

            "source": "https://medlineplus.gov/diabetestype2.html",

            "content": (
                "A1C measures average blood glucose "
                "over the past three months."
            )
        }

    ]

    # Finish request
    log_entry = logger.finish_request(

        request_data=request,

        rewritten_query=(
            "Definition of A1C test "
            "in diabetes management"
        ),

        retrieved_documents=documents,

        answer=(
            "An A1C test is a blood test that "
            "measures average blood glucose "
            "over the past three months."
        )
    )

    print()

    print("Request logged successfully!")

    print()

    print("Request ID:")

    print(log_entry["request_id"])

    print()

    print("Latency:")

    print(
        f"{log_entry['latency_seconds']} seconds"
    )

    print()

    print("=" * 70)

    print("CURRENT MONITORING METRICS")

    print("=" * 70)

    metrics = MonitoringMetrics()

    metrics.print_metrics()


if __name__ == "__main__":

    main()