'use client';

import { useRef, useMemo, useEffect, useState } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import {
  OrbitControls,
  Sphere,
  Box,
  Torus,
  MeshDistortMaterial,
  Environment,
  Float,
  Trail,
  Text3D,
  Center,
  PerspectiveCamera,
  Stars
} from '@react-three/drei';
import * as THREE from 'three';
import { motion, AnimatePresence } from 'framer-motion';

// DNA 분자 구조 인터페이스
interface FragranceDNA {
  id: string;
  name: string;
  complexity: number;
  notes: {
    top: Array<{ name: string; intensity: number; color: string }>;
    heart: Array<{ name: string; intensity: number; color: string }>;
    base: Array<{ name: string; intensity: number; color: string }>;
  };
  molecularStructure: {
    bonds: number;
    rings: number;
    branches: number;
  };
  volatility: number;
  sillage: number;
  longevity: number;
  accords: Array<{ name: string; percentage: number }>;
}

// DNA 나선 구조 컴포넌트
function DNAHelix({ dna }: { dna: FragranceDNA }) {
  const groupRef = useRef<THREE.Group>(null);
  const [hovered, setHovered] = useState(false);

  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.rotation.y += 0.003;
      if (hovered) {
        groupRef.current.scale.lerp(new THREE.Vector3(1.1, 1.1, 1.1), 0.1);
      } else {
        groupRef.current.scale.lerp(new THREE.Vector3(1, 1, 1), 0.1);
      }
    }
  });

  // DNA 나선의 점들 생성
  const helixPoints = useMemo(() => {
    const points: Array<{ pos: [number, number, number]; color: string; size: number }> = [];
    const height = 8;
    const radius = 2;
    const turns = 3;
    const pointsPerTurn = 20;

    for (let i = 0; i < turns * pointsPerTurn; i++) {
      const t = i / (turns * pointsPerTurn);
      const angle = t * Math.PI * 2 * turns;
      const y = (t - 0.5) * height;

      // 이중 나선 구조
      for (let strand = 0; strand < 2; strand++) {
        const phase = strand * Math.PI;
        const x = Math.cos(angle + phase) * radius;
        const z = Math.sin(angle + phase) * radius;

        // 노트 타입에 따른 색상 결정
        let color = '#8B6F47';
        let size = 0.15;

        if (t < 0.33) {
          // Top notes - 밝은 색상
          const noteIndex = Math.floor(t * 3 * dna.notes.top.length);
          if (dna.notes.top[noteIndex]) {
            color = dna.notes.top[noteIndex].color;
            size = 0.1 + dna.notes.top[noteIndex].intensity * 0.1;
          }
        } else if (t < 0.66) {
          // Heart notes - 중간 색상
          const noteIndex = Math.floor((t - 0.33) * 3 * dna.notes.heart.length);
          if (dna.notes.heart[noteIndex]) {
            color = dna.notes.heart[noteIndex].color;
            size = 0.12 + dna.notes.heart[noteIndex].intensity * 0.12;
          }
        } else {
          // Base notes - 진한 색상
          const noteIndex = Math.floor((t - 0.66) * 3 * dna.notes.base.length);
          if (dna.notes.base[noteIndex]) {
            color = dna.notes.base[noteIndex].color;
            size = 0.15 + dna.notes.base[noteIndex].intensity * 0.15;
          }
        }

        points.push({ pos: [x, y, z], color, size });
      }

      // 연결선 (수평 연결)
      if (i % 3 === 0) {
        const x1 = Math.cos(angle) * radius;
        const z1 = Math.sin(angle) * radius;
        const x2 = Math.cos(angle + Math.PI) * radius;
        const z2 = Math.sin(angle + Math.PI) * radius;

        // 중간 지점들
        for (let j = 0.25; j <= 0.75; j += 0.25) {
          points.push({
            pos: [x1 * (1 - j) + x2 * j, y, z1 * (1 - j) + z2 * j],
            color: '#FFD700',
            size: 0.08
          });
        }
      }
    }

    return points;
  }, [dna]);

  return (
    <group
      ref={groupRef}
      onPointerOver={() => setHovered(true)}
      onPointerOut={() => setHovered(false)}
    >
      {/* DNA 점들 */}
      {helixPoints.map((point, i) => (
        <Float key={i} speed={2} rotationIntensity={0.5} floatIntensity={0.3}>
          <Sphere position={point.pos} args={[point.size, 16, 16]}>
            <meshPhysicalMaterial
              color={point.color}
              emissive={point.color}
              emissiveIntensity={0.5}
              metalness={0.8}
              roughness={0.2}
              clearcoat={1}
              clearcoatRoughness={0}
            />
          </Sphere>
        </Float>
      ))}

      {/* 중심축 */}
      <Box position={[0, 0, 0]} args={[0.05, 8, 0.05]}>
        <meshPhysicalMaterial
          color="#FFD700"
          emissive="#FFD700"
          emissiveIntensity={0.3}
          metalness={1}
          roughness={0}
          opacity={0.3}
          transparent
        />
      </Box>

      {/* 복잡도 링 */}
      {Array.from({ length: Math.floor(dna.complexity) }, (_, i) => (
        <Torus
          key={i}
          position={[0, (i - dna.complexity / 2) * 0.8, 0]}
          args={[2.5, 0.03, 8, 32]}
          rotation={[Math.PI / 2, 0, i * 0.3]}
        >
          <meshPhysicalMaterial
            color="#8B6F47"
            emissive="#8B6F47"
            emissiveIntensity={0.2}
            metalness={0.9}
            roughness={0.1}
            opacity={0.2}
            transparent
          />
        </Torus>
      ))}
    </group>
  );
}

// 향료 분자 구조
function MolecularStructure({ structure }: { structure: FragranceDNA['molecularStructure'] }) {
  const groupRef = useRef<THREE.Group>(null);

  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.rotation.x = Math.sin(state.clock.elapsedTime * 0.5) * 0.1;
      groupRef.current.rotation.y += 0.005;
    }
  });

  return (
    <group ref={groupRef} position={[5, 0, 0]}>
      {/* 분자 결합 */}
      {Array.from({ length: structure.bonds }, (_, i) => {
        const angle = (i / structure.bonds) * Math.PI * 2;
        const radius = 1.5;
        return (
          <group key={`bond-${i}`}>
            <Sphere
              position={[
                Math.cos(angle) * radius,
                Math.sin(i * 0.5) * 0.5,
                Math.sin(angle) * radius
              ]}
              args={[0.2, 16, 16]}
            >
              <meshPhysicalMaterial
                color="#FF6B6B"
                emissive="#FF6B6B"
                emissiveIntensity={0.5}
                metalness={0.7}
                roughness={0.3}
              />
            </Sphere>
            <Box
              position={[
                Math.cos(angle) * radius * 0.5,
                Math.sin(i * 0.5) * 0.25,
                Math.sin(angle) * radius * 0.5
              ]}
              args={[radius, 0.05, 0.05]}
              rotation={[0, angle, 0]}
            >
              <meshPhysicalMaterial
                color="#4ECDC4"
                opacity={0.6}
                transparent
              />
            </Box>
          </group>
        );
      })}

      {/* 고리 구조 */}
      {Array.from({ length: structure.rings }, (_, i) => (
        <Torus
          key={`ring-${i}`}
          position={[0, (i - structure.rings / 2) * 0.5, 0]}
          args={[1, 0.1, 6, 16]}
          rotation={[i * 0.3, 0, 0]}
        >
          <meshPhysicalMaterial
            color="#95E1D3"
            emissive="#95E1D3"
            emissiveIntensity={0.3}
            metalness={0.8}
            roughness={0.2}
          />
        </Torus>
      ))}
    </group>
  );
}

// 향수 특성 시각화
function FragranceCharacteristics({ dna }: { dna: FragranceDNA }) {
  return (
    <group position={[-5, 0, 0]}>
      {/* Volatility (휘발성) */}
      <Float speed={dna.volatility * 2} floatIntensity={dna.volatility}>
        <Sphere position={[0, 2, 0]} args={[0.5, 32, 32]}>
          <MeshDistortMaterial
            color="#FFE66D"
            emissive="#FFE66D"
            emissiveIntensity={0.5}
            distort={dna.volatility * 0.5}
            speed={2}
            metalness={0.5}
            roughness={0.3}
          />
        </Sphere>
      </Float>

      {/* Sillage (확산력) */}
      <group position={[0, 0, 0]}>
        {Array.from({ length: Math.floor(dna.sillage * 5) }, (_, i) => (
          <Sphere
            key={i}
            position={[
              Math.cos((i / 5) * Math.PI * 2) * (1 + i * 0.3),
              0,
              Math.sin((i / 5) * Math.PI * 2) * (1 + i * 0.3)
            ]}
            args={[0.1 + i * 0.02, 16, 16]}
          >
            <meshPhysicalMaterial
              color="#A8DADC"
              opacity={1 - i * 0.15}
              transparent
              emissive="#A8DADC"
              emissiveIntensity={0.3}
            />
          </Sphere>
        ))}
      </group>

      {/* Longevity (지속력) */}
      <Box position={[0, -2, 0]} args={[0.5, dna.longevity * 2, 0.5]}>
        <meshPhysicalMaterial
          color="#457B9D"
          emissive="#457B9D"
          emissiveIntensity={0.4}
          metalness={0.7}
          roughness={0.2}
          clearcoat={1}
        />
      </Box>
    </group>
  );
}

// 메인 DNA 뷰어 컴포넌트
export default function FragranceDNAViewer({ dna }: { dna?: FragranceDNA }) {
  const [showDetails, setShowDetails] = useState(true);
  const [selectedAccord, setSelectedAccord] = useState<string | null>(null);

  // 샘플 DNA 데이터 (실제로는 props로 받음)
  const defaultDNA: FragranceDNA = {
    id: 'DNA-2025-001',
    name: 'Ethereal Dreams',
    complexity: 8.5,
    notes: {
      top: [
        { name: 'Bergamot', intensity: 0.9, color: '#FFE66D' },
        { name: 'Lemon', intensity: 0.7, color: '#FFF59D' },
        { name: 'Pink Pepper', intensity: 0.5, color: '#FF8A80' }
      ],
      heart: [
        { name: 'Rose', intensity: 0.8, color: '#FF6B9D' },
        { name: 'Jasmine', intensity: 0.9, color: '#C06C84' },
        { name: 'Iris', intensity: 0.6, color: '#9C88FF' }
      ],
      base: [
        { name: 'Sandalwood', intensity: 0.85, color: '#8B6F47' },
        { name: 'Amber', intensity: 0.9, color: '#FFA726' },
        { name: 'Musk', intensity: 0.7, color: '#795548' }
      ]
    },
    molecularStructure: {
      bonds: 12,
      rings: 3,
      branches: 8
    },
    volatility: 0.7,
    sillage: 0.8,
    longevity: 0.9,
    accords: [
      { name: 'Floral', percentage: 35 },
      { name: 'Woody', percentage: 25 },
      { name: 'Citrus', percentage: 20 },
      { name: 'Amber', percentage: 15 },
      { name: 'Spicy', percentage: 5 }
    ]
  };

  const currentDNA = dna || defaultDNA;

  return (
    <div className="w-full h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-violet-900 relative">
      {/* 3D Canvas */}
      <Canvas>
        <PerspectiveCamera makeDefault position={[0, 0, 15]} />
        <OrbitControls
          enablePan={true}
          enableZoom={true}
          enableRotate={true}
          maxDistance={30}
          minDistance={5}
        />

        {/* 조명 */}
        <ambientLight intensity={0.3} />
        <pointLight position={[10, 10, 10]} intensity={0.8} />
        <pointLight position={[-10, -10, -10]} intensity={0.5} color="#8B6F47" />
        <spotLight
          position={[0, 10, 0]}
          angle={0.5}
          penumbra={1}
          intensity={0.5}
          color="#FFD700"
        />

        {/* 배경 별 */}
        <Stars
          radius={100}
          depth={50}
          count={5000}
          factor={4}
          saturation={0}
          fade
          speed={1}
        />

        {/* DNA 나선 */}
        <DNAHelix dna={currentDNA} />

        {/* 분자 구조 */}
        <MolecularStructure structure={currentDNA.molecularStructure} />

        {/* 향수 특성 */}
        <FragranceCharacteristics dna={currentDNA} />

        {/* 환경 */}
        <Environment preset="night" />
      </Canvas>

      {/* UI 오버레이 */}
      <div className="absolute top-0 left-0 right-0 bottom-0 pointer-events-none">
        {/* 헤더 */}
        <div className="absolute top-8 left-8 right-8 pointer-events-auto">
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-black/30 backdrop-blur-md rounded-2xl p-6 border border-white/10"
          >
            <h1 className="text-3xl font-bold text-white mb-2">{currentDNA.name}</h1>
            <p className="text-gray-300">DNA ID: {currentDNA.id}</p>
            <div className="flex gap-4 mt-4">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-green-400 rounded-full animate-pulse"></div>
                <span className="text-sm text-gray-300">실시간 분석 중</span>
              </div>
              <div className="text-sm text-gray-300">
                복잡도: {currentDNA.complexity}/10
              </div>
            </div>
          </motion.div>
        </div>

        {/* 사이드 패널 - 노트 정보 */}
        <AnimatePresence>
          {showDetails && (
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              className="absolute right-8 top-32 w-80 pointer-events-auto"
            >
              <div className="bg-black/30 backdrop-blur-md rounded-2xl p-6 border border-white/10">
                <h2 className="text-xl font-semibold text-white mb-4">향료 구성</h2>

                {/* Top Notes */}
                <div className="mb-4">
                  <h3 className="text-sm font-medium text-gray-400 mb-2">탑 노트</h3>
                  <div className="space-y-2">
                    {currentDNA.notes.top.map((note) => (
                      <div key={note.name} className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <div
                            className="w-3 h-3 rounded-full"
                            style={{ backgroundColor: note.color }}
                          />
                          <span className="text-white text-sm">{note.name}</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <div className="w-20 bg-gray-700 rounded-full h-2">
                            <div
                              className="h-2 rounded-full bg-gradient-to-r from-yellow-400 to-yellow-600"
                              style={{ width: `${note.intensity * 100}%` }}
                            />
                          </div>
                          <span className="text-xs text-gray-400">
                            {Math.round(note.intensity * 100)}%
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Heart Notes */}
                <div className="mb-4">
                  <h3 className="text-sm font-medium text-gray-400 mb-2">하트 노트</h3>
                  <div className="space-y-2">
                    {currentDNA.notes.heart.map((note) => (
                      <div key={note.name} className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <div
                            className="w-3 h-3 rounded-full"
                            style={{ backgroundColor: note.color }}
                          />
                          <span className="text-white text-sm">{note.name}</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <div className="w-20 bg-gray-700 rounded-full h-2">
                            <div
                              className="h-2 rounded-full bg-gradient-to-r from-pink-400 to-pink-600"
                              style={{ width: `${note.intensity * 100}%` }}
                            />
                          </div>
                          <span className="text-xs text-gray-400">
                            {Math.round(note.intensity * 100)}%
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Base Notes */}
                <div className="mb-4">
                  <h3 className="text-sm font-medium text-gray-400 mb-2">베이스 노트</h3>
                  <div className="space-y-2">
                    {currentDNA.notes.base.map((note) => (
                      <div key={note.name} className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <div
                            className="w-3 h-3 rounded-full"
                            style={{ backgroundColor: note.color }}
                          />
                          <span className="text-white text-sm">{note.name}</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <div className="w-20 bg-gray-700 rounded-full h-2">
                            <div
                              className="h-2 rounded-full bg-gradient-to-r from-amber-600 to-amber-800"
                              style={{ width: `${note.intensity * 100}%` }}
                            />
                          </div>
                          <span className="text-xs text-gray-400">
                            {Math.round(note.intensity * 100)}%
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* 특성 카드 */}
              <div className="bg-black/30 backdrop-blur-md rounded-2xl p-6 border border-white/10 mt-4">
                <h2 className="text-xl font-semibold text-white mb-4">향수 특성</h2>
                <div className="space-y-3">
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-400">휘발성</span>
                      <span className="text-white">{(currentDNA.volatility * 100).toFixed(0)}%</span>
                    </div>
                    <div className="w-full bg-gray-700 rounded-full h-2">
                      <div
                        className="h-2 rounded-full bg-gradient-to-r from-yellow-400 to-orange-500"
                        style={{ width: `${currentDNA.volatility * 100}%` }}
                      />
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-400">확산력</span>
                      <span className="text-white">{(currentDNA.sillage * 100).toFixed(0)}%</span>
                    </div>
                    <div className="w-full bg-gray-700 rounded-full h-2">
                      <div
                        className="h-2 rounded-full bg-gradient-to-r from-cyan-400 to-blue-500"
                        style={{ width: `${currentDNA.sillage * 100}%` }}
                      />
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-400">지속력</span>
                      <span className="text-white">{(currentDNA.longevity * 100).toFixed(0)}%</span>
                    </div>
                    <div className="w-full bg-gray-700 rounded-full h-2">
                      <div
                        className="h-2 rounded-full bg-gradient-to-r from-purple-400 to-purple-600"
                        style={{ width: `${currentDNA.longevity * 100}%` }}
                      />
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* 하단 Accords 바 */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="absolute bottom-8 left-8 right-8 pointer-events-auto"
        >
          <div className="bg-black/30 backdrop-blur-md rounded-2xl p-6 border border-white/10">
            <h2 className="text-lg font-semibold text-white mb-4">향조 구성</h2>
            <div className="flex gap-3">
              {currentDNA.accords.map((accord) => (
                <button
                  key={accord.name}
                  onClick={() => setSelectedAccord(accord.name)}
                  className={`px-4 py-2 rounded-lg border transition-all ${
                    selectedAccord === accord.name
                      ? 'bg-white/20 border-white/40 text-white'
                      : 'bg-white/5 border-white/20 text-gray-300 hover:bg-white/10'
                  }`}
                >
                  <span className="font-medium">{accord.name}</span>
                  <span className="ml-2 text-sm opacity-70">{accord.percentage}%</span>
                </button>
              ))}
            </div>
          </div>
        </motion.div>

        {/* 컨트롤 버튼 */}
        <button
          onClick={() => setShowDetails(!showDetails)}
          className="absolute top-8 right-8 p-3 bg-white/10 backdrop-blur-md rounded-lg border border-white/20 text-white hover:bg-white/20 transition-all pointer-events-auto"
        >
          {showDetails ? '상세 정보 숨기기' : '상세 정보 보기'}
        </button>
      </div>
    </div>
  );
}