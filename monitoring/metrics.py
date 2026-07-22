import json
from pathlib import Path
from statistics import mean


class MonitoringMetrics:

    def __init__(
        self,
        log_file="monitoring/logs/requests.jsonl"
    ):

        self.log_file = Path(log_file)

    def load_logs(self):

        if not self.log_file.exists():

            return []

        logs = []

        with open(
            self.log_file,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                try:

                    logs.append(
                        json.loads(line)
                    )

                except json.JSONDecodeError:

                    continue

        return logs

    def calculate_metrics(self):

        logs = self.load_logs()

        if not logs:

            return {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "success_rate": 0,
                "average_latency": 0,
                "average_documents_retrieved": 0
            }

        total_requests = len(logs)

        successful_requests = sum(
            1
            for log in logs
            if log.get("success") is True
        )

        failed_requests = total_requests - successful_requests

        latencies = [
            log.get("latency_seconds", 0)
            for log in logs
            if log.get("latency_seconds") is not None
        ]

        documents = [
            log.get("num_documents", 0)
            for log in logs
        ]

        return {

            "total_requests": total_requests,

            "successful_requests": successful_requests,

            "failed_requests": failed_requests,

            "success_rate": round(
                successful_requests / total_requests,
                4
            ),

            "average_latency": round(
                mean(latencies),
                4
            ) if latencies else 0,

            "average_documents_retrieved": round(
                mean(documents),
                2
            ) if documents else 0
        }

    def print_metrics(self):

        metrics = self.calculate_metrics()

        print("=" * 70)

        print("MEDICAL KNOWLEDGE AI - MONITORING METRICS")

        print("=" * 70)

        print()

        print(
            f"Total Requests: "
            f"{metrics['total_requests']}"
        )

        print(
            f"Successful Requests: "
            f"{metrics['successful_requests']}"
        )

        print(
            f"Failed Requests: "
            f"{metrics['failed_requests']}"
        )

        print(
            f"Success Rate: "
            f"{metrics['success_rate'] * 100:.2f}%"
        )

        print(
            f"Average Latency: "
            f"{metrics['average_latency']:.2f} seconds"
        )

        print(
            f"Average Documents Retrieved: "
            f"{metrics['average_documents_retrieved']:.2f}"
        )

        print()

        print("=" * 70)


if __name__ == "__main__":

    monitoring = MonitoringMetrics()

    monitoring.print_metrics()