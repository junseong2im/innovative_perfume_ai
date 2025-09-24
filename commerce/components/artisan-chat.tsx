'use client';

import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  recipePreview?: any;
}

export default function ArtisanChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [conversationId, setConversationId] = useState<string>('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';

  useEffect(() => {
    // 부드러운 스크롤
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    // 웰컴 메시지
    if (messages.length === 0) {
      setTimeout(() => {
        setMessages([{
          id: 'welcome',
          role: 'assistant',
          content: '안녕하세요. 저는 Artisan, 당신의 개인 향수 아티스트입니다. 어떤 향을 찾고 계신가요?',
          timestamp: new Date()
        }]);
      }, 500);
    }
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);
    setIsTyping(true);

    try {
      const response = await fetch(`${API_URL}/api/v1/generate/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: input,
          conversation_id: conversationId,
          history: messages.map(m => ({
            role: m.role,
            content: m.content
          }))
        })
      });

      const data = await response.json();

      // 타이핑 효과
      setIsTyping(false);

      const assistantMessage: Message = {
        id: Date.now().toString(),
        role: 'assistant',
        content: data.response,
        timestamp: new Date(),
        recipePreview: data.recipe_preview
      };

      setMessages(prev => [...prev, assistantMessage]);
      setConversationId(data.conversation_id);

    } catch (error) {
      console.error('Error:', error);
      setIsTyping(false);
      setMessages(prev => [...prev, {
        id: Date.now().toString(),
        role: 'assistant',
        content: '죄송합니다. 일시적인 오류가 발생했습니다. 다시 시도해 주세요.',
        timestamp: new Date()
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e as any);
    }
  };

  return (
    <div className="flex flex-col h-[600px] bg-white/80 backdrop-blur-sm rounded-2xl shadow-2xl overflow-hidden">
      {/* Header */}
      <div className="px-6 py-4 border-b border-gray-100 bg-gradient-to-r from-white to-gray-50">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-display text-gray-900">Artisan AI</h3>
            <p className="text-xs text-gray-500 mt-0.5">Your Personal Perfume Artist</p>
          </div>
          <div className="flex items-center space-x-1">
            <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
            <span className="text-xs text-gray-500">Online</span>
          </div>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-6 space-y-4">
        <AnimatePresence>
          {messages.map((message) => (
            <motion.div
              key={message.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.3 }}
              className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div className={`max-w-[80%] ${message.role === 'user' ? 'order-2' : 'order-1'}`}>
                <div
                  className={`px-4 py-3 rounded-2xl ${
                    message.role === 'user'
                      ? 'bg-gradient-to-r from-gray-800 to-gray-900 text-white'
                      : 'bg-gradient-to-r from-gray-50 to-gray-100 text-gray-800'
                  }`}
                >
                  <p className="text-sm leading-relaxed whitespace-pre-wrap">
                    {message.content}
                  </p>
                </div>

                {/* Recipe Preview */}
                {message.recipePreview && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    className="mt-3 p-4 bg-gradient-to-br from-purple-50 to-pink-50 rounded-xl"
                  >
                    <h4 className="font-display text-lg text-gray-900 mb-2">
                      {message.recipePreview.name}
                    </h4>
                    <p className="text-xs text-gray-600 mb-3">
                      {message.recipePreview.description}
                    </p>
                    <div className="space-y-1 text-xs">
                      <div className="flex items-start">
                        <span className="text-gray-500 w-16">Top:</span>
                        <span className="text-gray-700">{message.recipePreview.key_notes?.top?.join(', ')}</span>
                      </div>
                      <div className="flex items-start">
                        <span className="text-gray-500 w-16">Heart:</span>
                        <span className="text-gray-700">{message.recipePreview.key_notes?.heart?.join(', ')}</span>
                      </div>
                      <div className="flex items-start">
                        <span className="text-gray-500 w-16">Base:</span>
                        <span className="text-gray-700">{message.recipePreview.key_notes?.base?.join(', ')}</span>
                      </div>
                    </div>
                  </motion.div>
                )}

                <div className="mt-1 px-1">
                  <span className="text-xs text-gray-400">
                    {message.timestamp.toLocaleTimeString('ko-KR', {
                      hour: '2-digit',
                      minute: '2-digit'
                    })}
                  </span>
                </div>
              </div>
            </motion.div>
          ))}
        </AnimatePresence>

        {/* Typing Indicator */}
        {isTyping && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex justify-start"
          >
            <div className="bg-gray-100 rounded-2xl px-4 py-3">
              <div className="flex space-x-2">
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
              </div>
            </div>
          </motion.div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <form onSubmit={handleSubmit} className="border-t border-gray-100 p-4 bg-white">
        <div className="flex items-end space-x-3">
          <div className="flex-1 relative">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="어떤 향수를 만들어 드릴까요?"
              className="w-full px-4 py-3 bg-gray-50 border border-gray-200 rounded-xl resize-none focus:outline-none focus:border-gray-400 transition-colors text-sm"
              rows={1}
              style={{
                minHeight: '48px',
                maxHeight: '120px'
              }}
              disabled={isLoading}
            />
          </div>
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            type="submit"
            disabled={!input.trim() || isLoading}
            className="px-6 py-3 bg-gradient-to-r from-gray-800 to-gray-900 text-white rounded-xl font-medium text-sm disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg hover:shadow-xl"
          >
            {isLoading ? (
              <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
            ) : (
              <span>전송</span>
            )}
          </motion.button>
        </div>

        {/* Quick Actions */}
        <div className="mt-3 flex flex-wrap gap-2">
          {[
            '봄에 어울리는 향수',
            '로맨틱한 저녁 향수',
            '상쾌한 시트러스',
            '우디 오리엔탈'
          ].map((suggestion) => (
            <button
              key={suggestion}
              type="button"
              onClick={() => setInput(suggestion)}
              className="px-3 py-1.5 text-xs bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-full transition-colors"
            >
              {suggestion}
            </button>
          ))}
        </div>
      </form>
    </div>
  );
}