"""
Incident Management System
장애 등급(Sev1~Sev3), 15분 내 초기 대응, 블레임리스 포스트모템
"""

import time
import json
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum


class Severity(str, Enum):
    """장애 등급"""
    SEV1 = "Sev1"  # Critical - 서비스 완전 중단
    SEV2 = "Sev2"  # High - 주요 기능 장애
    SEV3 = "Sev3"  # Medium - 부분적 기능 장애
    SEV4 = "Sev4"  # Low - 경미한 문제


class IncidentStatus(str, Enum):
    """사건 상태"""
    OPEN = "open"
    INVESTIGATING = "investigating"
    IDENTIFIED = "identified"
    MITIGATING = "mitigating"
    RESOLVED = "resolved"
    CLOSED = "closed"


@dataclass
class IncidentTimeline:
    """사건 타임라인"""
    timestamp: str
    action: str
    description: str
    responder: Optional[str] = None


@dataclass
class Incident:
    """사건 정보"""
    incident_id: str
    title: str
    description: str
    severity: Severity
    status: IncidentStatus
    created_at: str
    detected_at: str
    acknowledged_at: Optional[str] = None
    mitigated_at: Optional[str] = None
    resolved_at: Optional[str] = None
    closed_at: Optional[str] = None

    # Response times
    time_to_acknowledge_minutes: Optional[float] = None
    time_to_mitigate_minutes: Optional[float] = None
    time_to_resolve_minutes: Optional[float] = None

    # Incident details
    affected_components: List[str] = field(default_factory=list)
    impacted_users: Optional[int] = None
    root_cause: Optional[str] = None
    resolution: Optional[str] = None

    # Team
    responders: List[str] = field(default_factory=list)
    incident_commander: Optional[str] = None

    # Timeline
    timeline: List[IncidentTimeline] = field(default_factory=list)

    # Postmortem
    postmortem_required: bool = True
    postmortem_completed: bool = False
    postmortem_link: Optional[str] = None


class IncidentManager:
    """사건 관리 시스템"""

    # Response time targets
    RESPONSE_TARGETS = {
        Severity.SEV1: 15,   # 15 minutes
        Severity.SEV2: 30,   # 30 minutes
        Severity.SEV3: 60,   # 60 minutes
        Severity.SEV4: 240   # 4 hours
    }

    def __init__(self, data_file: str = "incidents.json"):
        self.data_file = Path(data_file)
        self.incidents: Dict[str, Incident] = {}
        self._load_incidents()

    def _load_incidents(self):
        """사건 로드"""
        if self.data_file.exists():
            with open(self.data_file, 'r') as f:
                data = json.load(f)
                self.incidents = {
                    incident_id: Incident(**incident_data)
                    for incident_id, incident_data in data.items()
                }
        else:
            self.incidents = {}

    def _save_incidents(self):
        """사건 저장"""
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            incident_id: asdict(incident)
            for incident_id, incident in self.incidents.items()
        }
        with open(self.data_file, 'w') as f:
            json.dump(data, f, indent=2)

    def create_incident(
        self,
        title: str,
        description: str,
        severity: Severity,
        affected_components: List[str],
        detected_at: Optional[str] = None
    ) -> Incident:
        """사건 생성"""
        incident_id = f"INC-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        now = datetime.now().isoformat()
        detected = detected_at or now

        incident = Incident(
            incident_id=incident_id,
            title=title,
            description=description,
            severity=severity,
            status=IncidentStatus.OPEN,
            created_at=now,
            detected_at=detected,
            affected_components=affected_components,
            postmortem_required=severity in (Severity.SEV1, Severity.SEV2)
        )

        # Add creation to timeline
        incident.timeline.append(IncidentTimeline(
            timestamp=now,
            action="created",
            description=f"Incident created: {title}"
        ))

        self.incidents[incident_id] = incident
        self._save_incidents()

        print(f"🚨 Incident Created: {incident_id} [{severity}]")
        print(f"   Title: {title}")
        print(f"   Affected: {', '.join(affected_components)}")

        return incident

    def acknowledge_incident(
        self,
        incident_id: str,
        responder: str
    ):
        """사건 인지 (15분 내 목표)"""
        if incident_id not in self.incidents:
            raise ValueError(f"Incident not found: {incident_id}")

        incident = self.incidents[incident_id]
        now = datetime.now().isoformat()

        incident.acknowledged_at = now
        incident.status = IncidentStatus.INVESTIGATING
        incident.responders.append(responder)

        if not incident.incident_commander:
            incident.incident_commander = responder

        # Calculate time to acknowledge
        detected_time = datetime.fromisoformat(incident.detected_at)
        ack_time = datetime.fromisoformat(now)
        tta_minutes = (ack_time - detected_time).total_seconds() / 60
        incident.time_to_acknowledge_minutes = tta_minutes

        # Add to timeline
        incident.timeline.append(IncidentTimeline(
            timestamp=now,
            action="acknowledged",
            description=f"Incident acknowledged by {responder}",
            responder=responder
        ))

        # Check if response time target met
        target = self.RESPONSE_TARGETS[incident.severity]
        if tta_minutes > target:
            print(f"⚠️  Response time exceeded: {tta_minutes:.1f}min (target: {target}min)")
        else:
            print(f"✅ Response time met: {tta_minutes:.1f}min (target: {target}min)")

        self._save_incidents()

    def update_status(
        self,
        incident_id: str,
        status: IncidentStatus,
        description: str,
        responder: Optional[str] = None
    ):
        """사건 상태 업데이트"""
        if incident_id not in self.incidents:
            raise ValueError(f"Incident not found: {incident_id}")

        incident = self.incidents[incident_id]
        now = datetime.now().isoformat()

        old_status = incident.status
        incident.status = status

        # Update timestamps
        if status == IncidentStatus.MITIGATING and not incident.mitigated_at:
            incident.mitigated_at = now
            detected_time = datetime.fromisoformat(incident.detected_at)
            mitigated_time = datetime.fromisoformat(now)
            incident.time_to_mitigate_minutes = (mitigated_time - detected_time).total_seconds() / 60

        elif status == IncidentStatus.RESOLVED and not incident.resolved_at:
            incident.resolved_at = now
            detected_time = datetime.fromisoformat(incident.detected_at)
            resolved_time = datetime.fromisoformat(now)
            incident.time_to_resolve_minutes = (resolved_time - detected_time).total_seconds() / 60

        elif status == IncidentStatus.CLOSED and not incident.closed_at:
            incident.closed_at = now

        # Add to timeline
        incident.timeline.append(IncidentTimeline(
            timestamp=now,
            action=f"status_changed_{old_status}_to_{status}",
            description=description,
            responder=responder
        ))

        self._save_incidents()

    def add_timeline_entry(
        self,
        incident_id: str,
        action: str,
        description: str,
        responder: Optional[str] = None
    ):
        """타임라인 항목 추가"""
        if incident_id not in self.incidents:
            raise ValueError(f"Incident not found: {incident_id}")

        incident = self.incidents[incident_id]
        now = datetime.now().isoformat()

        incident.timeline.append(IncidentTimeline(
            timestamp=now,
            action=action,
            description=description,
            responder=responder
        ))

        self._save_incidents()

    def set_root_cause(
        self,
        incident_id: str,
        root_cause: str,
        responder: Optional[str] = None
    ):
        """근본 원인 설정"""
        if incident_id not in self.incidents:
            raise ValueError(f"Incident not found: {incident_id}")

        incident = self.incidents[incident_id]
        incident.root_cause = root_cause
        incident.status = IncidentStatus.IDENTIFIED

        self.add_timeline_entry(
            incident_id,
            "root_cause_identified",
            f"Root cause identified: {root_cause}",
            responder
        )

    def set_resolution(
        self,
        incident_id: str,
        resolution: str,
        responder: Optional[str] = None
    ):
        """해결 방법 설정"""
        if incident_id not in self.incidents:
            raise ValueError(f"Incident not found: {incident_id}")

        incident = self.incidents[incident_id]
        incident.resolution = resolution
        incident.status = IncidentStatus.RESOLVED

        self.add_timeline_entry(
            incident_id,
            "resolution_applied",
            f"Resolution applied: {resolution}",
            responder
        )

    def close_incident(
        self,
        incident_id: str,
        responder: Optional[str] = None
    ):
        """사건 종료"""
        if incident_id not in self.incidents:
            raise ValueError(f"Incident not found: {incident_id}")

        incident = self.incidents[incident_id]

        if incident.postmortem_required and not incident.postmortem_completed:
            print(f"⚠️  Warning: Postmortem required but not completed for {incident_id}")

        self.update_status(
            incident_id,
            IncidentStatus.CLOSED,
            "Incident closed",
            responder
        )

    def complete_postmortem(
        self,
        incident_id: str,
        postmortem_link: str
    ):
        """포스트모템 완료"""
        if incident_id not in self.incidents:
            raise ValueError(f"Incident not found: {incident_id}")

        incident = self.incidents[incident_id]
        incident.postmortem_completed = True
        incident.postmortem_link = postmortem_link

        self.add_timeline_entry(
            incident_id,
            "postmortem_completed",
            f"Postmortem completed: {postmortem_link}"
        )

    def get_incident(self, incident_id: str) -> Optional[Incident]:
        """사건 조회"""
        return self.incidents.get(incident_id)

    def get_open_incidents(self) -> List[Incident]:
        """열린 사건 목록"""
        return [
            incident for incident in self.incidents.values()
            if incident.status not in (IncidentStatus.RESOLVED, IncidentStatus.CLOSED)
        ]

    def get_incidents_by_severity(self, severity: Severity) -> List[Incident]:
        """등급별 사건 목록"""
        return [
            incident for incident in self.incidents.values()
            if incident.severity == severity
        ]

    def get_response_time_report(self) -> Dict[str, Dict]:
        """응답 시간 리포트"""
        report = {}

        for severity in Severity:
            incidents = self.get_incidents_by_severity(severity)
            if not incidents:
                continue

            ack_times = [
                i.time_to_acknowledge_minutes
                for i in incidents
                if i.time_to_acknowledge_minutes is not None
            ]

            target = self.RESPONSE_TARGETS[severity]

            report[severity.value] = {
                "count": len(incidents),
                "target_minutes": target,
                "avg_ack_time": sum(ack_times) / len(ack_times) if ack_times else None,
                "max_ack_time": max(ack_times) if ack_times else None,
                "target_met_percentage": sum(1 for t in ack_times if t <= target) / len(ack_times) * 100 if ack_times else 0
            }

        return report

    def print_incident_report(self):
        """사건 리포트 출력"""
        print("=" * 80)
        print("Incident Report")
        print("=" * 80)
        print()

        # Open incidents
        open_incidents = self.get_open_incidents()
        print(f"Open Incidents: {len(open_incidents)}")
        for incident in open_incidents:
            severity_emoji = {
                Severity.SEV1: "🔴",
                Severity.SEV2: "🟠",
                Severity.SEV3: "🟡",
                Severity.SEV4: "🟢"
            }.get(incident.severity, "⚪")

            print(f"  {severity_emoji} [{incident.severity}] {incident.incident_id}: {incident.title}")
            print(f"     Status: {incident.status} | Created: {incident.created_at}")

        print()

        # Response time report
        print("Response Time Performance:")
        response_report = self.get_response_time_report()
        for severity, data in response_report.items():
            target_met = data["target_met_percentage"]
            emoji = "✅" if target_met >= 95 else "⚠️" if target_met >= 80 else "🔴"

            print(f"  {emoji} {severity}: {data['count']} incidents")
            print(f"     Target: {data['target_minutes']}min | Avg: {data['avg_ack_time']:.1f}min | Met: {target_met:.1f}%")

        print()

        # Postmortem status
        postmortem_required = [
            i for i in self.incidents.values()
            if i.postmortem_required and i.status == IncidentStatus.CLOSED
        ]
        postmortem_pending = [i for i in postmortem_required if not i.postmortem_completed]

        print(f"Postmortems: {len(postmortem_required)} required, {len(postmortem_pending)} pending")
        if postmortem_pending:
            for incident in postmortem_pending[:5]:
                print(f"  📝 {incident.incident_id}: {incident.title}")

        print()


# =============================================================================
# Global Instance
# =============================================================================

_incident_manager: Optional[IncidentManager] = None


def get_incident_manager() -> IncidentManager:
    """글로벌 사건 관리자 반환"""
    global _incident_manager
    if _incident_manager is None:
        _incident_manager = IncidentManager()
    return _incident_manager


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Incident Manager")
    parser.add_argument("--report", action="store_true", help="Show incident report")
    parser.add_argument("--create", help="Create incident (JSON)")
    parser.add_argument("--ack", help="Acknowledge incident ID")
    parser.add_argument("--responder", default="oncall", help="Responder name")

    args = parser.parse_args()

    manager = get_incident_manager()

    if args.report:
        manager.print_incident_report()
    elif args.create:
        # Example: --create '{"title":"API Down","description":"...","severity":"Sev1","components":["API"]}'
        data = json.loads(args.create)
        incident = manager.create_incident(
            title=data["title"],
            description=data["description"],
            severity=Severity(data["severity"]),
            affected_components=data["components"]
        )
        print(f"Created: {incident.incident_id}")
    elif args.ack:
        manager.acknowledge_incident(args.ack, args.responder)
        print(f"Acknowledged: {args.ack}")
    else:
        manager.print_incident_report()
