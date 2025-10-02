"""
실제 핵심 엔진 테스트
PostgreSQL 기반 도구들이 제대로 작동하는지 확인
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("="*80)
print("핵심 엔진 실제 작동 테스트 - PostgreSQL 기반")
print("="*80)


def test_database_connection():
    """데이터베이스 연결 테스트"""
    print("\n[TEST 1] 데이터베이스 연결")
    print("-"*40)

    try:
        from fragrance_ai.database.connection import DatabaseManager

        db_manager = DatabaseManager()

        # 연결 테스트
        if db_manager.test_connection():
            print("✓ PostgreSQL 연결 성공")
        else:
            print("✗ PostgreSQL 연결 실패")
            return False

        # pgvector 확인
        if db_manager.check_pgvector():
            print("✓ pgvector 확장 활성화")
        else:
            print("⚠ pgvector 미설치 (벡터 검색 제한)")

        # 테이블 확인
        with db_manager.get_session() as session:
            from fragrance_ai.database.schema import Note, Fragrance, BlendingRule

            note_count = session.query(Note).count()
            fragrance_count = session.query(Fragrance).count()
            rule_count = session.query(BlendingRule).count()

            print(f"✓ 데이터 로드 확인:")
            print(f"  - 노트: {note_count}개")
            print(f"  - 향수: {fragrance_count}개")
            print(f"  - 규칙: {rule_count}개")

        return True

    except Exception as e:
        print(f"✗ 데이터베이스 테스트 실패: {e}")
        return False


def test_hybrid_search():
    """하이브리드 검색 테스트"""
    print("\n[TEST 2] HybridSearchTool - 실제 벡터 검색")
    print("-"*40)

    try:
        from fragrance_ai.tools.hybrid_search_tool import HybridSearchTool

        search_tool = HybridSearchTool()

        # 1. 향수 검색
        print("향수 검색: 'romantic floral'")
        results = search_tool.search(
            query="romantic floral",
            search_type="fragrance",
            top_k=3
        )

        if results:
            print(f"✓ {len(results)}개 결과 발견:")
            for r in results[:2]:
                print(f"  - {r.name} (유사도: {r.similarity:.2f})")
        else:
            print("⚠ 검색 결과 없음")

        # 2. 노트 검색
        print("\n노트 검색: 'citrus fresh'")
        results = search_tool.search(
            query="citrus fresh",
            search_type="note",
            top_k=3
        )

        if results:
            print(f"✓ {len(results)}개 노트 발견:")
            for r in results[:2]:
                print(f"  - {r.name}: {r.description[:50]}...")
        else:
            print("⚠ 노트 검색 결과 없음")

        # 3. 지식 검색
        print("\n지식 검색: 'perfume history'")
        results = search_tool.search(
            query="perfume history",
            search_type="knowledge",
            top_k=2
        )

        if results:
            print(f"✓ {len(results)}개 지식 발견")
        else:
            print("⚠ 지식 검색 결과 없음")

        print("\n[PASS] HybridSearchTool 작동 확인")
        return True

    except Exception as e:
        print(f"[FAIL] 검색 도구 오류: {e}")
        return False


def test_validator_tool():
    """검증 도구 테스트"""
    print("\n[TEST 3] ValidatorTool - 데이터베이스 기반 검증")
    print("-"*40)

    try:
        from fragrance_ai.tools.validator_tool import ScientificValidator, NotesComposition

        validator = ScientificValidator()

        # 캐시 확인
        if validator._notes_cache:
            print(f"✓ {len(validator._notes_cache)}개 노트 캐싱됨")
        if validator._blending_rules_cache:
            print(f"✓ {len(validator._blending_rules_cache)}개 규칙 캐싱됨")

        # 테스트 조합
        test_composition = NotesComposition(
            top_notes=[{"Bergamot": 15.0}, {"Lemon": 10.0}],
            heart_notes=[{"Rose": 20.0}, {"Jasmine": 15.0}],
            base_notes=[{"Sandalwood": 20.0}, {"Musk": 10.0}, {"Vanilla": 10.0}]
        )

        # 검증 수행
        result = validator.validate(test_composition)

        print(f"\n검증 결과:")
        print(f"  - 유효성: {'✓' if result.is_valid else '✗'}")
        print(f"  - 조화도: {result.harmony_score:.1f}/10")
        print(f"  - 안정성: {result.stability_score:.1f}/10")
        print(f"  - 지속성: {result.longevity_score:.1f}/10")
        print(f"  - 확산성: {result.sillage_score:.1f}/10")
        print(f"  - 종합점수: {result.overall_score:.1f}/10")
        print(f"  - 신뢰도: {result.confidence:.1%}")

        if result.key_risks:
            print(f"\n위험 요소:")
            for risk in result.key_risks[:2]:
                print(f"  - {risk}")

        if result.suggestions:
            print(f"\n개선 제안:")
            for sug in result.suggestions[:2]:
                print(f"  - {sug}")

        print("\n[PASS] ValidatorTool 작동 확인")
        return True

    except Exception as e:
        print(f"[FAIL] 검증 도구 오류: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_knowledge_tool():
    """지식베이스 도구 테스트"""
    print("\n[TEST 4] KnowledgeBaseTool - PostgreSQL 지식 검색")
    print("-"*40)

    try:
        from fragrance_ai.tools.knowledge_tool import PerfumeKnowledgeBase, KnowledgeQuery

        kb = PerfumeKnowledgeBase()

        # 1. 기술 질의
        query1 = KnowledgeQuery(
            category="technique",
            query="distillation process"
        )
        response1 = kb.query(query1)

        print(f"기술 질의 결과:")
        print(f"  - 답변: {response1.answer[:100]}...")
        print(f"  - 신뢰도: {response1.confidence:.1%}")
        print(f"  - 출처: {', '.join(response1.sources)}")

        # 2. 노트 질의
        query2 = KnowledgeQuery(
            category="note",
            query="bergamot"
        )
        response2 = kb.query(query2)

        print(f"\n노트 질의 결과:")
        print(f"  - 답변 길이: {len(response2.answer)}자")
        print(f"  - 신뢰도: {response2.confidence:.1%}")

        # 3. 역사 질의
        query3 = KnowledgeQuery(
            category="history",
            query="ancient egypt perfume"
        )
        response3 = kb.query(query3)

        print(f"\n역사 질의 결과:")
        print(f"  - 답변: {response3.answer[:80]}...")
        print(f"  - 관련 주제: {', '.join(response3.related_topics[:3])}")

        print("\n[PASS] KnowledgeBaseTool 작동 확인")
        return True

    except Exception as e:
        print(f"[FAIL] 지식베이스 도구 오류: {e}")
        return False


def test_moga_with_db():
    """MOGA와 데이터베이스 통합 테스트"""
    print("\n[TEST 5] MOGA + Database 통합")
    print("-"*40)

    try:
        from fragrance_ai.training.moga_optimizer_real import RealMOGAOptimizer
        from fragrance_ai.tools.hybrid_search_tool import HybridSearchTool

        # MOGA 최적화
        moga = RealMOGAOptimizer(
            population_size=20,
            generations=5,
            gene_size=8
        )

        print("MOGA 최적화 실행...")
        pareto_front = moga.optimize()

        if pareto_front:
            best = pareto_front[0]
            recipe = moga.individual_to_recipe(best)

            print(f"✓ 최적 레시피 생성:")
            print(f"  - Top: {len(recipe['top_notes'])}개")
            print(f"  - Middle: {len(recipe['middle_notes'])}개")
            print(f"  - Base: {len(recipe['base_notes'])}개")
            print(f"  - Fitness: {best.fitness.values}")

            # 생성된 DNA를 데이터베이스 검색
            search_tool = HybridSearchTool()
            similar_dnas = search_tool.search_similar_dna(
                dna_sequence=best,
                top_k=3
            )

            if similar_dnas:
                print(f"\n✓ 유사 DNA {len(similar_dnas)}개 발견")
            else:
                print("\n⚠ 유사 DNA 없음 (새로운 조합)")

            print("\n[PASS] MOGA-DB 통합 작동")
            return True
        else:
            print("[FAIL] MOGA 최적화 실패")
            return False

    except Exception as e:
        print(f"[FAIL] MOGA-DB 통합 오류: {e}")
        return False


def main():
    """메인 테스트 실행"""
    print("\n테스트 시작...")
    print("주의: PostgreSQL이 실행 중이어야 합니다.")
    print("데이터베이스가 없으면: python setup_database.py")
    print("="*80)

    results = {}

    # 1. 데이터베이스 연결
    results["Database"] = test_database_connection()

    if results["Database"]:
        # 2. 검색 도구
        results["HybridSearch"] = test_hybrid_search()

        # 3. 검증 도구
        results["Validator"] = test_validator_tool()

        # 4. 지식베이스
        results["Knowledge"] = test_knowledge_tool()

        # 5. MOGA 통합
        results["MOGA-DB"] = test_moga_with_db()
    else:
        print("\n⚠ 데이터베이스 연결 실패로 나머지 테스트 건너뜀")
        results["HybridSearch"] = False
        results["Validator"] = False
        results["Knowledge"] = False
        results["MOGA-DB"] = False

    # 최종 결과
    print("\n" + "="*80)
    print("테스트 결과 요약")
    print("="*80)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} - {test_name}")

    all_passed = all(results.values())

    if all_passed:
        print("\n🎉 모든 핵심 엔진이 실제로 작동합니다!")
        print("PostgreSQL 기반 실제 AI 시스템 구축 완료")
    else:
        print("\n❌ 일부 테스트 실패")
        print("setup_database.py를 먼저 실행하세요")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)