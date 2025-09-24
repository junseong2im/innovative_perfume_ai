'use client';

import { useState, useEffect } from 'react';
import { motion, useScroll, useTransform } from 'framer-motion';
import ArtisanChat from '@/components/artisan-chat';
import Image from 'next/image';

export default function ArtisanPage() {
  const [activeSection, setActiveSection] = useState('hero');
  const { scrollYProgress } = useScroll();
  const opacity = useTransform(scrollYProgress, [0, 0.5], [1, 0]);
  const scale = useTransform(scrollYProgress, [0, 0.5], [1, 1.1]);

  useEffect(() => {
    const handleScroll = () => {
      const sections = ['hero', 'philosophy', 'artisan', 'experience'];
      const scrollPosition = window.scrollY + window.innerHeight / 2;

      for (const section of sections) {
        const element = document.getElementById(section);
        if (element) {
          const { offsetTop, offsetHeight } = element;
          if (scrollPosition >= offsetTop && scrollPosition < offsetTop + offsetHeight) {
            setActiveSection(section);
            break;
          }
        }
      }
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <div className="relative">
      {/* Navigation Dots */}
      <div className="fixed right-8 top-1/2 -translate-y-1/2 z-50 hidden lg:flex flex-col space-y-4">
        {['hero', 'philosophy', 'artisan', 'experience'].map((section) => (
          <button
            key={section}
            onClick={() => document.getElementById(section)?.scrollIntoView({ behavior: 'smooth' })}
            className={`w-3 h-3 rounded-full transition-all duration-300 ${
              activeSection === section
                ? 'bg-gray-900 scale-125'
                : 'bg-gray-300 hover:bg-gray-600'
            }`}
            aria-label={`Navigate to ${section}`}
          />
        ))}
      </div>

      {/* Hero Section */}
      <section id="hero" className="relative h-screen flex items-center justify-center overflow-hidden">
        <motion.div
          style={{ opacity, scale }}
          className="absolute inset-0 z-0"
        >
          <div className="absolute inset-0 bg-gradient-to-b from-black/40 via-black/20 to-black/40 z-10" />
          <video
            autoPlay
            muted
            loop
            playsInline
            className="w-full h-full object-cover"
          >
            <source src="/videos/luxury-perfume.mp4" type="video/mp4" />
          </video>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1.2, ease: 'easeOut' }}
          className="relative z-20 text-center text-white px-4"
        >
          <motion.h1
            className="font-display text-6xl md:text-8xl font-light tracking-wider mb-6"
            initial={{ letterSpacing: '0.5em', opacity: 0 }}
            animate={{ letterSpacing: '0.2em', opacity: 1 }}
            transition={{ duration: 1.5, delay: 0.3 }}
          >
            ARTISAN
          </motion.h1>

          <motion.p
            className="text-xl md:text-2xl font-light tracking-wide mb-8"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 1, delay: 0.8 }}
          >
            Your Personal AI Perfumer
          </motion.p>

          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 1, delay: 1.2 }}
          >
            <button
              onClick={() => document.getElementById('artisan')?.scrollIntoView({ behavior: 'smooth' })}
              className="group relative px-8 py-4 overflow-hidden"
            >
              <span className="relative z-10 text-sm tracking-widest font-light">
                BEGIN YOUR JOURNEY
              </span>
              <div className="absolute inset-0 border border-white/50 group-hover:border-white transition-colors duration-500" />
              <div className="absolute inset-0 bg-white transform scale-x-0 group-hover:scale-x-100 transition-transform duration-500 origin-left" />
              <span className="absolute inset-0 flex items-center justify-center text-gray-900 transform scale-x-0 group-hover:scale-x-100 transition-transform duration-500 origin-left text-sm tracking-widest font-light">
                BEGIN YOUR JOURNEY
              </span>
            </button>
          </motion.div>
        </motion.div>

        {/* Scroll Indicator */}
        <motion.div
          className="absolute bottom-8 left-1/2 -translate-x-1/2"
          animate={{ y: [0, 10, 0] }}
          transition={{ duration: 2, repeat: Infinity }}
        >
          <div className="w-6 h-10 border-2 border-white/50 rounded-full flex justify-center">
            <div className="w-1 h-3 bg-white/50 rounded-full mt-2" />
          </div>
        </motion.div>
      </section>

      {/* Philosophy Section */}
      <section id="philosophy" className="min-h-screen flex items-center py-24 px-8">
        <div className="max-w-7xl mx-auto grid md:grid-cols-2 gap-16 items-center">
          <motion.div
            initial={{ opacity: 0, x: -50 }}
            whileInView={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
          >
            <h2 className="font-display text-5xl md:text-6xl mb-8 text-gray-900">
              The Art of
              <br />
              <span className="italic">Personalization</span>
            </h2>
            <p className="text-lg text-gray-600 leading-relaxed mb-6">
              향수는 단순한 향이 아닙니다. 그것은 당신의 이야기, 당신의 감정,
              그리고 당신만의 독특한 정체성을 표현하는 예술입니다.
            </p>
            <p className="text-lg text-gray-600 leading-relaxed">
              Artisan은 최첨단 AI 기술과 전통적인 조향 예술을 결합하여,
              당신만을 위한 완벽한 향수를 창조합니다. 각각의 레시피는 유니크하며,
              당신의 개성과 취향을 섬세하게 반영합니다.
            </p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: 50 }}
            whileInView={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="relative h-[600px]"
          >
            <div className="absolute inset-0 bg-gradient-to-br from-purple-100 to-pink-100 rounded-2xl" />
            <Image
              src="/images/artisan-philosophy.jpg"
              alt="Artisan Philosophy"
              fill
              className="object-cover rounded-2xl"
            />
          </motion.div>
        </div>
      </section>

      {/* Artisan Chat Section */}
      <section id="artisan" className="min-h-screen bg-gradient-to-b from-gray-50 to-white py-24 px-8">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="font-display text-5xl md:text-6xl mb-6 text-gray-900">
              Meet Your Artisan
            </h2>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              대화를 통해 당신만의 시그니처 향수를 창조하세요.
              Artisan이 당신의 이야기를 향기로 번역해드립니다.
            </p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            whileInView={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="max-w-4xl mx-auto"
          >
            <ArtisanChat />
          </motion.div>

          {/* Features Grid */}
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.3 }}
            viewport={{ once: true }}
            className="grid md:grid-cols-3 gap-8 mt-20"
          >
            {[
              {
                title: 'Personalized Creation',
                description: '당신의 개성과 선호도를 반영한 맞춤형 향수 레시피',
                icon: '✨'
              },
              {
                title: 'Scientific Validation',
                description: '딥러닝 기반 과학적 검증으로 완벽한 조화 보장',
                icon: '🔬'
              },
              {
                title: 'Artisan Expertise',
                description: '전문 조향사의 지식과 AI의 창의성이 만나는 곳',
                icon: '🎨'
              }
            ].map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                viewport={{ once: true }}
                className="text-center p-8 bg-white rounded-2xl shadow-lg hover:shadow-xl transition-shadow"
              >
                <div className="text-4xl mb-4">{feature.icon}</div>
                <h3 className="font-display text-2xl mb-3 text-gray-900">{feature.title}</h3>
                <p className="text-gray-600">{feature.description}</p>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* Experience Section */}
      <section id="experience" className="relative min-h-screen flex items-center justify-center overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-gray-900 via-gray-800 to-black" />

        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          transition={{ duration: 1.5 }}
          viewport={{ once: true }}
          className="relative z-10 text-center text-white px-8"
        >
          <h2 className="font-display text-5xl md:text-7xl mb-8">
            Begin Your
            <br />
            <span className="italic">Fragrance Journey</span>
          </h2>

          <p className="text-xl md:text-2xl font-light mb-12 max-w-2xl mx-auto">
            세상에 단 하나뿐인 당신만의 향수를 만들어보세요.
            Artisan과 함께라면 가능합니다.
          </p>

          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => document.getElementById('artisan')?.scrollIntoView({ behavior: 'smooth' })}
            className="px-12 py-5 bg-white text-gray-900 font-medium tracking-widest hover:bg-gray-100 transition-colors"
          >
            START CREATING
          </motion.button>
        </motion.div>
      </section>
    </div>
  );
}