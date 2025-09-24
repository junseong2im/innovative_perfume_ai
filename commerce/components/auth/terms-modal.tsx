'use client';

import { useState } from 'react';

interface TermsModalProps {
  isOpen: boolean;
  onClose: () => void;
  type: 'terms' | 'privacy';
}

export default function TermsModal({ isOpen, onClose, type }: TermsModalProps) {
  if (!isOpen) return null;

  const content = {
    terms: {
      title: '이용약관',
      content: `
제1조 (목적)
이 약관은 들숨(Deulsoom) 회사(이하 "회사")가 운영하는 웹사이트에서 제공하는 인터넷 관련 서비스(이하 "서비스")를 이용함에 있어 사이버몰과 이용자의 권리, 의무 및 책임사항을 규정함을 목적으로 합니다.

제2조 (정의)
1. "몰"이란 회사가 재화 또는 용역(이하 "재화등")을 이용자에게 제공하기 위하여 컴퓨터등 정보통신설비를 이용하여 재화등을 거래할 수 있도록 설정한 가상의 영업장을 말하며, 아울러 사이버몰을 운영하는 사업자의 의미로도 사용합니다.
2. "이용자"란 "몰"에 접속하여 이 약관에 따라 "몰"이 제공하는 서비스를 받는 회원 및 비회원을 말합니다.

제3조 (약관의 명시와 설명 및 개정)
1. "몰"은 이 약관의 내용과 상호 및 대표자 성명, 영업소 소재지 주소(소비자의 불만을 처리할 수 있는 곳의 주소를 포함), 전화번호, 모사전송번호, 전자우편주소, 사업자등록번호, 통신판매업 신고번호, 개인정보관리책임자등을 이용자가 쉽게 알 수 있도록 들숨 사이버몰의 초기 서비스화면(전면)에 게시합니다.

제4조 (서비스의 제공 및 변경)
1. "몰"은 다음과 같은 업무를 수행합니다.
   - 재화 또는 용역에 대한 정보 제공 및 구매계약의 체결
   - 구매계약이 체결된 재화 또는 용역의 배송
   - 기타 "몰"이 정하는 업무

제5조 (서비스의 중단)
1. "몰"은 컴퓨터 등 정보통신설비의 보수점검, 교체 및 고장, 통신의 두절 등의 사유가 발생한 경우에는 서비스의 제공을 일시적으로 중단할 수 있습니다.

제6조 (회원가입)
1. 이용자는 "몰"이 정한 가입 양식에 따라 회원정보를 기입한 후 이 약관에 동의한다는 의사표시를 함으로서 회원가입을 신청합니다.
2. "몰"은 제1항과 같이 회원으로 가입할 것을 신청한 이용자 중 다음 각호에 해당하지 않는 한 회원으로 등록합니다.

제7조 (회원 탈퇴 및 자격 상실 등)
1. 회원은 "몰"에 언제든지 탈퇴를 요청할 수 있으며 "몰"은 즉시 회원탈퇴를 처리합니다.

제8조 (개인정보보호)
1. "몰"은 이용자의 개인정보 수집시 서비스제공을 위하여 필요한 범위에서 최소한의 개인정보를 수집합니다.
2. "몰"은 회원가입시 구매계약이행에 필요한 정보를 미리 수집하지 않습니다.

본 약관은 2024년 1월 1일부터 시행됩니다.
      `
    },
    privacy: {
      title: '개인정보 처리방침',
      content: `
들숨(Deulsoom)은 개인정보보호법에 따라 이용자의 개인정보 보호 및 권익을 보호하고 개인정보와 관련한 이용자의 고충을 원활하게 처리할 수 있도록 다음과 같은 처리방침을 두고 있습니다.

1. 개인정보의 처리목적
들숨은 다음의 목적을 위하여 개인정보를 처리하고 있으며, 다음의 목적 이외의 용도로는 이용하지 않습니다.
- 고객 가입의사 확인, 고객에 대한 서비스 제공에 따른 본인 식별·인증, 회원자격 유지·관리, 제품 및 서비스 공급에 따른 금액 결제, 제품 및 서비스의 공급·배송 등

2. 개인정보의 처리 및 보유기간
① 들숨은 정보주체로부터 개인정보를 수집할 때 동의받은 개인정보 보유·이용기간 또는 법령에 따른 개인정보 보유·이용기간 내에서 개인정보를 처리·보유합니다.
② 구체적인 개인정보 처리 및 보유 기간은 다음과 같습니다.
- 고객 가입 및 관리: 서비스 이용계약 또는 회원가입 해지시까지, 다만 채권·채무관계 잔존시에는 해당 채권·채무관계 정산시까지

3. 개인정보의 제3자 제공
① 들숨은 정보주체의 개인정보를 1항에서 명시한 범위 내에서만 처리하며, 정보주체의 동의, 법률의 특별한 규정 등 개인정보보호법 제17조에 해당하는 경우에만 개인정보를 제3자에게 제공합니다.

4. 개인정보처리의 위탁
① 들숨은 원활한 개인정보 업무처리를 위하여 다음과 같이 개인정보 처리업무를 위탁하고 있습니다.
- 위탁받는 자: 결제대행업체
- 위탁하는 업무의 내용: 결제처리 및 배송업무

5. 정보주체의 권리·의무 및 행사방법
이용자는 개인정보주체로서 다음과 같은 권리를 행사할 수 있습니다.
① 개인정보 처리현황 통지요구
② 개인정보 처리정지 요구
③ 개인정보의 수정·삭제 요구
④ 손해배상 청구

6. 처리하는 개인정보 항목
들숨은 다음의 개인정보 항목을 처리하고 있습니다.
- 필수항목: 이메일, 비밀번호, 이름, 휴대전화번호
- 선택항목: 생년월일, 성별

7. 개인정보의 파기
들숨은 원칙적으로 개인정보 처리목적이 달성된 경우에는 지체없이 해당 개인정보를 파기합니다.

8. 개인정보 보호책임자
① 들숨은 개인정보 처리에 관한 업무를 총괄해서 책임지고, 개인정보 처리와 관련한 정보주체의 불만처리 및 피해구제 등을 위하여 아래와 같이 개인정보 보호책임자를 지정하고 있습니다.

▶ 개인정보 보호책임자
성명: 개인정보관리팀
직책: 팀장
연락처: privacy@deulsoom.com

본 방침은 2024년 1월 1일부터 시행됩니다.
      `
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div
        className="absolute inset-0 bg-black bg-opacity-50"
        onClick={onClose}
      />

      <div className="relative bg-white rounded-xl shadow-2xl max-w-4xl w-full max-h-[80vh] overflow-hidden">
        <div className="flex items-center justify-between p-6 border-b" style={{ borderColor: 'var(--vintage-gray-light)' }}>
          <h2 className="text-2xl font-light" style={{ color: 'var(--vintage-navy)' }}>
            {content[type].title}
          </h2>
          <button
            onClick={onClose}
            className="p-2 rounded-lg transition-colors"
            style={{ color: 'var(--vintage-gray)' }}
            onMouseEnter={(e) => e.target.style.color = 'var(--vintage-navy)'}
            onMouseLeave={(e) => e.target.style.color = 'var(--vintage-gray)'}
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="p-6 overflow-y-auto max-h-[60vh]">
          <div className="prose prose-sm max-w-none" style={{ color: 'var(--vintage-navy)' }}>
            <pre className="whitespace-pre-wrap font-sans text-sm leading-relaxed">
              {content[type].content}
            </pre>
          </div>
        </div>

        <div className="p-6 border-t bg-gray-50" style={{ borderColor: 'var(--vintage-gray-light)' }}>
          <button
            onClick={onClose}
            className="w-full py-3 px-6 rounded-lg font-medium transition-all"
            style={{
              backgroundColor: 'var(--vintage-gold)',
              color: 'white'
            }}
            onMouseEnter={(e) => e.target.style.backgroundColor = 'var(--vintage-gold-dark)'}
            onMouseLeave={(e) => e.target.style.backgroundColor = 'var(--vintage-gold)'}
          >
            확인
          </button>
        </div>
      </div>
    </div>
  );
}