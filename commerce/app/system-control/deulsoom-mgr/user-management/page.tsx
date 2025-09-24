'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import UserService, { User } from '../../../../lib/user-service';

interface AdminUser {
  id: string;
  username: string;
  role: 'super_admin' | 'admin' | 'marketing_admin' | 'dev_admin';
  loginTime: string;
  sessionToken: string;
}


interface NewUser {
  username: string;
  email: string;
  password: string;
  role: string;
}

export default function UserManagement() {
  const router = useRouter();
  const [adminUser, setAdminUser] = useState<AdminUser | null>(null);
  const [users, setUsers] = useState<User[]>([]);
  const [filteredUsers, setFilteredUsers] = useState<User[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [roleFilter, setRoleFilter] = useState('all');
  const [statusFilter, setStatusFilter] = useState('all');
  const [showAddUser, setShowAddUser] = useState(false);
  const [editingUser, setEditingUser] = useState<User | null>(null);
  const [newUser, setNewUser] = useState<NewUser>({
    username: '',
    email: '',
    password: '',
    role: 'customer'
  });
  const userService = UserService.getInstance();

  useEffect(() => {
    // 관리자 인증 확인
    const adminData = localStorage.getItem('deulsoom_admin_secure');
    if (!adminData) {
      router.push('/system-control/deulsoom-mgr');
      return;
    }

    const user = JSON.parse(adminData);
    setAdminUser(user);

    // 권한 확인 - 사용자 관리는 super_admin과 admin만 가능
    if (user.role !== 'super_admin' && user.role !== 'admin') {
      alert('사용자 관리 권한이 없습니다.');
      router.push('/system-control/deulsoom-mgr/dashboard');
      return;
    }

    // 통합 사용자 서비스에서 데이터 로드
    loadUsers();
  }, [router]);

  const loadUsers = () => {
    // 통합 사용자 서비스에서 데이터 가져오기
    const allUsers = userService.getAllUsers();
    setUsers(allUsers);
    setFilteredUsers(allUsers);

    // 사용자 데이터 변경 감지
    const handleUsersUpdate = (event: CustomEvent) => {
      const updatedUsers = event.detail;
      setUsers(updatedUsers);
    };

    window.addEventListener('usersUpdated', handleUsersUpdate as EventListener);

    return () => {
      window.removeEventListener('usersUpdated', handleUsersUpdate as EventListener);
    };
  };

  useEffect(() => {
    // 통합 서비스의 검색 기능 사용
    const filtered = userService.searchUsers(searchTerm, roleFilter, statusFilter);
    setFilteredUsers(filtered);
  }, [users, searchTerm, roleFilter, statusFilter, userService]);

  const handleAddUser = () => {
    if (!newUser.username || !newUser.email || !newUser.password) {
      alert('모든 필드를 입력해주세요.');
      return;
    }

    // 통합 서비스를 통해 사용자 추가
    const addedUser = userService.addUser({
      username: newUser.username,
      email: newUser.email,
      role: newUser.role as any,
      status: 'active'
    });

    // UI 업데이트
    setUsers(prev => [...prev, addedUser]);
    setNewUser({ username: '', email: '', password: '', role: 'customer' });
    setShowAddUser(false);
  };

  const handleEditUser = (user: User) => {
    setEditingUser(user);
  };

  const handleUpdateUser = () => {
    if (!editingUser) return;

    // 통합 서비스를 통해 사용자 업데이트
    const updatedUser = userService.updateUser(editingUser.id, editingUser);
    if (updatedUser) {
      setUsers(prev => prev.map(user =>
        user.id === editingUser.id ? updatedUser : user
      ));
    }
    setEditingUser(null);
  };

  const handleDeleteUser = (userId: string) => {
    if (confirm('정말로 이 사용자를 삭제하시겠습니까?')) {
      // 통합 서비스를 통해 사용자 삭제
      const success = userService.deleteUser(userId);
      if (success) {
        setUsers(prev => prev.filter(user => user.id !== userId));
      } else {
        alert('관리자 계정은 삭제할 수 없습니다.');
      }
    }
  };

  const handleStatusChange = (userId: string, status: 'active' | 'inactive' | 'suspended') => {
    // 통합 서비스를 통해 상태 변경
    const success = userService.updateUserStatus(userId, status);
    if (success) {
      setUsers(prev => prev.map(user =>
        user.id === userId ? { ...user, status } : user
      ));
    }
  };

  const getRoleDisplayName = (role: string) => {
    const roleNames = {
      'customer': '고객',
      'admin': '관리자',
      'super_admin': '최고 관리자',
      'marketing_admin': '마케팅 관리자',
      'dev_admin': '개발자'
    };
    return roleNames[role as keyof typeof roleNames] || role;
  };

  const getStatusColor = (status: string) => {
    const colors = {
      'active': 'bg-green-100 text-green-800',
      'inactive': 'bg-gray-100 text-gray-800',
      'suspended': 'bg-red-100 text-red-800'
    };
    return colors[status as keyof typeof colors] || 'bg-gray-100 text-gray-800';
  };

  if (!adminUser) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-900">
        <div className="text-white">로딩 중...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900">
      {/* 헤더 */}
      <header className="bg-black border-b border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-4">
              <Link href="/system-control/deulsoom-mgr/dashboard" className="text-blue-400 hover:text-blue-300">
                ← 대시보드
              </Link>
              <h1 className="text-xl font-semibold text-white">사용자 관리</h1>
            </div>

            <div className="flex items-center space-x-4">
              <span className="text-gray-300">{adminUser.username}</span>
              <Link
                href="/system-control/deulsoom-mgr"
                className="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-md text-sm transition-colors"
              >
                로그아웃
              </Link>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* 검색 및 필터 */}
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-6 mb-6">
          <div className="flex flex-wrap gap-4 items-center justify-between">
            <div className="flex flex-wrap gap-4 items-center">
              <input
                type="text"
                placeholder="사용자명 또는 이메일 검색..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />

              <select
                value={roleFilter}
                onChange={(e) => setRoleFilter(e.target.value)}
                className="px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="all">모든 역할</option>
                <option value="customer">고객</option>
                <option value="admin">관리자</option>
                <option value="super_admin">최고 관리자</option>
                <option value="marketing_admin">마케팅 관리자</option>
                <option value="dev_admin">개발자</option>
              </select>

              <select
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
                className="px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="all">모든 상태</option>
                <option value="active">활성</option>
                <option value="inactive">비활성</option>
                <option value="suspended">정지</option>
              </select>
            </div>

            <button
              onClick={() => setShowAddUser(true)}
              className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition-colors"
            >
              사용자 추가
            </button>
          </div>
        </div>

        {/* 사용자 목록 */}
        <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-700">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">사용자</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">역할</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">상태</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">가입일</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">최근 접속</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">주문/매출</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">작업</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-700">
                {filteredUsers.map((user) => (
                  <tr key={user.id} className="hover:bg-gray-750">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div>
                        <div className="text-sm font-medium text-white">{user.username}</div>
                        <div className="text-sm text-gray-400">{user.email}</div>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className="text-sm text-gray-300">{getRoleDisplayName(user.role)}</span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${getStatusColor(user.status)}`}>
                        {user.status === 'active' ? '활성' : user.status === 'inactive' ? '비활성' : '정지'}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">
                      {user.createdAt}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">
                      {user.lastLogin}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">
                      {user.orders}건 / ₩{user.totalSpent.toLocaleString()}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm space-x-2">
                      <button
                        onClick={() => handleEditUser(user)}
                        className="text-blue-400 hover:text-blue-300"
                      >
                        편집
                      </button>
                      <select
                        value={user.status}
                        onChange={(e) => handleStatusChange(user.id, e.target.value as any)}
                        className="bg-gray-700 border border-gray-600 rounded text-white text-xs px-2 py-1"
                      >
                        <option value="active">활성</option>
                        <option value="inactive">비활성</option>
                        <option value="suspended">정지</option>
                      </select>
                      {user.role === 'customer' && (
                        <button
                          onClick={() => handleDeleteUser(user.id)}
                          className="text-red-400 hover:text-red-300"
                        >
                          삭제
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* 통계 */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mt-8">
          <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
            <h3 className="text-lg font-medium text-white mb-2">전체 사용자</h3>
            <p className="text-3xl font-bold text-blue-400">{users.length}</p>
          </div>
          <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
            <h3 className="text-lg font-medium text-white mb-2">활성 사용자</h3>
            <p className="text-3xl font-bold text-green-400">{users.filter(u => u.status === 'active').length}</p>
          </div>
          <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
            <h3 className="text-lg font-medium text-white mb-2">관리자 계정</h3>
            <p className="text-3xl font-bold text-purple-400">{users.filter(u => u.role !== 'customer').length}</p>
          </div>
          <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
            <h3 className="text-lg font-medium text-white mb-2">정지된 계정</h3>
            <p className="text-3xl font-bold text-red-400">{users.filter(u => u.status === 'suspended').length}</p>
          </div>
        </div>
      </div>

      {/* 사용자 추가 모달 */}
      {showAddUser && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-gray-800 rounded-lg border border-gray-700 p-6 w-full max-w-md mx-4">
            <h2 className="text-xl font-semibold text-white mb-4">새 사용자 추가</h2>

            <div className="space-y-4">
              <input
                type="text"
                placeholder="사용자명"
                value={newUser.username}
                onChange={(e) => setNewUser({...newUser, username: e.target.value})}
                className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />

              <input
                type="email"
                placeholder="이메일"
                value={newUser.email}
                onChange={(e) => setNewUser({...newUser, email: e.target.value})}
                className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />

              <input
                type="password"
                placeholder="비밀번호"
                value={newUser.password}
                onChange={(e) => setNewUser({...newUser, password: e.target.value})}
                className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />

              <select
                value={newUser.role}
                onChange={(e) => setNewUser({...newUser, role: e.target.value})}
                className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="customer">고객</option>
                {adminUser?.role === 'super_admin' && (
                  <>
                    <option value="admin">관리자</option>
                    <option value="marketing_admin">마케팅 관리자</option>
                    <option value="dev_admin">개발자</option>
                  </>
                )}
              </select>
            </div>

            <div className="flex justify-end space-x-3 mt-6">
              <button
                onClick={() => setShowAddUser(false)}
                className="px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg transition-colors"
              >
                취소
              </button>
              <button
                onClick={handleAddUser}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
              >
                추가
              </button>
            </div>
          </div>
        </div>
      )}

      {/* 사용자 편집 모달 */}
      {editingUser && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-gray-800 rounded-lg border border-gray-700 p-6 w-full max-w-md mx-4">
            <h2 className="text-xl font-semibold text-white mb-4">사용자 정보 편집</h2>

            <div className="space-y-4">
              <input
                type="text"
                placeholder="사용자명"
                value={editingUser.username}
                onChange={(e) => setEditingUser({...editingUser, username: e.target.value})}
                className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />

              <input
                type="email"
                placeholder="이메일"
                value={editingUser.email}
                onChange={(e) => setEditingUser({...editingUser, email: e.target.value})}
                className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />

              <select
                value={editingUser.role}
                onChange={(e) => setEditingUser({...editingUser, role: e.target.value as any})}
                className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                disabled={editingUser.role === 'super_admin' && adminUser?.role !== 'super_admin'}
              >
                <option value="customer">고객</option>
                {(adminUser?.role === 'super_admin' || editingUser.role === 'admin') && (
                  <option value="admin">관리자</option>
                )}
                {(adminUser?.role === 'super_admin' || editingUser.role === 'marketing_admin') && (
                  <option value="marketing_admin">마케팅 관리자</option>
                )}
                {(adminUser?.role === 'super_admin' || editingUser.role === 'dev_admin') && (
                  <option value="dev_admin">개발자</option>
                )}
                {adminUser?.role === 'super_admin' && (
                  <option value="super_admin">최고 관리자</option>
                )}
              </select>
            </div>

            <div className="flex justify-end space-x-3 mt-6">
              <button
                onClick={() => setEditingUser(null)}
                className="px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg transition-colors"
              >
                취소
              </button>
              <button
                onClick={handleUpdateUser}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
              >
                저장
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}