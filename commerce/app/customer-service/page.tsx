'use client';

import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Send, 
  Package, 
  Truck, 
  RefreshCw, 
  CreditCard,
  HelpCircle,
  User,
  MessageCircle,
  Star,
  ChevronDown,
  Phone,
  Mail,
  Clock
} from 'lucide-react';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  category?: string;
  sentiment?: string;
}

interface FAQ {
  category: string;
  questions: Array<{
    q: string;
    a: string;
  }>;
}

export default function CustomerService() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [showFeedback, setShowFeedback] = useState(false);
  const [rating, setRating] = useState(0);
  const [faqs, setFaqs] = useState<FAQ[]>([]);
  const [expandedFaq, setExpandedFaq] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Welcome message
  useEffect(() => {
    setMessages([{
      id: '1',
      role: 'assistant',
      content: "Hello! Welcome to Deulsoom Customer Service. I'm here to help with orders, shipping, returns, or any questions about our luxury fragrances. How may I assist you today?",
      timestamp: new Date()
    }]);
    
    // Load FAQs
    loadFAQs();
  }, []);

  // Auto-scroll to bottom
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const loadFAQs = async () => {
    try {
      const res = await fetch('/api/v1/customer-service/faq');
      const data = await res.json();
      if (data.success) {
        setFaqs(data.categories);
      }
    } catch (error) {
      console.error('Failed to load FAQs:', error);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await fetch('/api/v1/customer-service/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: input,
          session_id: sessionId
        })
      });

      const data = await response.json();

      if (data.success) {
        if (!sessionId) {
          setSessionId(data.session_id);
        }

        const assistantMessage: Message = {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: data.response,
          timestamp: new Date(),
          category: data.category?.category,
          sentiment: data.sentiment?.sentiment
        };

        setMessages(prev => [...prev, assistantMessage]);

        // Show feedback after 5 messages
        if (messages.length > 8 && !showFeedback) {
          setShowFeedback(true);
        }

        // Redirect to Artisan if needed
        if (data.redirect) {
          setTimeout(() => {
            if (confirm('Would you like to visit our Artisan AI Perfumer for fragrance creation?')) {
              window.location.href = data.redirect;
            }
          }, 1500);
        }
      }
    } catch (error) {
      console.error('Chat error:', error);
      setMessages(prev => [...prev, {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'I apologize for the technical issue. Please try again or contact support@deulsoom.com',
        timestamp: new Date()
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleRating = async (value: number) => {
    setRating(value);
    if (sessionId) {
      try {
        await fetch('/api/v1/customer-service/feedback', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            session_id: sessionId,
            rating: value,
            comment: value < 3 ? 'Could be better' : 'Good service'
          })
        });
        
        setTimeout(() => {
          setShowFeedback(false);
        }, 2000);
      } catch (error) {
        console.error('Feedback error:', error);
      }
    }
  };

  const quickActions = [
    { icon: Package, label: 'Track Order', query: 'I want to track my order' },
    { icon: Truck, label: 'Shipping Info', query: 'What are your shipping options?' },
    { icon: RefreshCw, label: 'Returns', query: 'How do I return an item?' },
    { icon: CreditCard, label: 'Payment', query: 'What payment methods do you accept?' },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <MessageCircle className="h-8 w-8 text-blue-600" />
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Customer Service</h1>
                <p className="text-sm text-gray-600">We're here to help</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-6 text-sm text-gray-600">
              <div className="flex items-center space-x-2">
                <Clock className="h-4 w-4" />
                <span>24/7 Support</span>
              </div>
              <div className="flex items-center space-x-2">
                <Phone className="h-4 w-4" />
                <span>1-800-DEULSOOM</span>
              </div>
              <div className="flex items-center space-x-2">
                <Mail className="h-4 w-4" />
                <span>support@deulsoom.com</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="container mx-auto px-4 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Chat Section */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-xl shadow-lg overflow-hidden">
              {/* Quick Actions */}
              <div className="p-4 border-b bg-gradient-to-r from-blue-50 to-indigo-50">
                <p className="text-sm font-semibold text-gray-700 mb-3">Quick Actions</p>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                  {quickActions.map((action, index) => (
                    <motion.button
                      key={index}
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={() => {
                        setInput(action.query);
                        inputRef.current?.focus();
                      }}
                      className="flex items-center justify-center space-x-2 p-2 bg-white rounded-lg border hover:border-blue-400 transition-colors"
                    >
                      <action.icon className="h-4 w-4 text-blue-600" />
                      <span className="text-xs font-medium">{action.label}</span>
                    </motion.button>
                  ))}
                </div>
              </div>

              {/* Messages */}
              <div className="h-[500px] overflow-y-auto p-4 space-y-4">
                <AnimatePresence initial={false}>
                  {messages.map((message) => (
                    <motion.div
                      key={message.id}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -20 }}
                      transition={{ duration: 0.3 }}
                      className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                    >
                      <div className={`max-w-[80%] ${message.role === 'user' ? 'order-2' : 'order-1'}`}>
                        <div className="flex items-start space-x-2">
                          {message.role === 'assistant' && (
                            <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center flex-shrink-0">
                              <MessageCircle className="h-4 w-4 text-white" />
                            </div>
                          )}
                          <div
                            className={`rounded-lg px-4 py-3 ${
                              message.role === 'user'
                                ? 'bg-blue-600 text-white'
                                : 'bg-gray-100 text-gray-800'
                            }`}
                          >
                            <p className="text-sm leading-relaxed">{message.content}</p>
                            <p className={`text-xs mt-1 ${
                              message.role === 'user' ? 'text-blue-200' : 'text-gray-500'
                            }`}>
                              {message.timestamp.toLocaleTimeString()}
                            </p>
                          </div>
                          {message.role === 'user' && (
                            <div className="w-8 h-8 rounded-full bg-gray-300 flex items-center justify-center flex-shrink-0">
                              <User className="h-4 w-4 text-gray-700" />
                            </div>
                          )}
                        </div>
                      </div>
                    </motion.div>
                  ))}
                </AnimatePresence>
                
                {loading && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="flex justify-start"
                  >
                    <div className="bg-gray-100 rounded-lg px-4 py-3">
                      <div className="flex space-x-2">
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-100" />
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-200" />
                      </div>
                    </div>
                  </motion.div>
                )}
                
                <div ref={messagesEndRef} />
              </div>

              {/* Feedback */}
              {showFeedback && rating === 0 && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  className="p-4 border-t bg-yellow-50"
                >
                  <p className="text-sm text-gray-700 mb-2">How was your experience?</p>
                  <div className="flex space-x-2">
                    {[1, 2, 3, 4, 5].map((value) => (
                      <button
                        key={value}
                        onClick={() => handleRating(value)}
                        className="hover:scale-110 transition-transform"
                      >
                        <Star
                          className={`h-6 w-6 ${
                            value <= rating ? 'text-yellow-400 fill-yellow-400' : 'text-gray-300'
                          }`}
                        />
                      </button>
                    ))}
                  </div>
                </motion.div>
              )}

              {/* Input */}
              <form onSubmit={handleSubmit} className="p-4 border-t bg-gray-50">
                <div className="flex space-x-2">
                  <input
                    ref={inputRef}
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Type your message..."
                    className="flex-1 px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                    disabled={loading}
                  />
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    type="submit"
                    disabled={loading || !input.trim()}
                    className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    <Send className="h-5 w-5" />
                  </motion.button>
                </div>
              </form>
            </div>
          </div>

          {/* FAQ Section */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-xl shadow-lg p-6">
              <div className="flex items-center space-x-2 mb-4">
                <HelpCircle className="h-5 w-5 text-blue-600" />
                <h2 className="text-lg font-bold text-gray-900">Frequently Asked</h2>
              </div>
              
              <div className="space-y-3">
                {faqs.map((faq, index) => (
                  <div key={index} className="border-b last:border-0 pb-3 last:pb-0">
                    <button
                      onClick={() => setExpandedFaq(expandedFaq === faq.category ? null : faq.category)}
                      className="w-full text-left flex items-center justify-between py-2 hover:text-blue-600 transition-colors"
                    >
                      <span className="font-medium text-sm">{faq.category}</span>
                      <ChevronDown 
                        className={`h-4 w-4 transition-transform ${
                          expandedFaq === faq.category ? 'rotate-180' : ''
                        }`}
                      />
                    </button>
                    
                    <AnimatePresence>
                      {expandedFaq === faq.category && (
                        <motion.div
                          initial={{ opacity: 0, height: 0 }}
                          animate={{ opacity: 1, height: 'auto' }}
                          exit={{ opacity: 0, height: 0 }}
                          className="space-y-2 mt-2"
                        >
                          {faq.questions.map((qa, qIndex) => (
                            <div key={qIndex} className="pl-4">
                              <button
                                onClick={() => {
                                  setInput(qa.q);
                                  inputRef.current?.focus();
                                }}
                                className="text-left hover:text-blue-600 transition-colors"
                              >
                                <p className="text-sm font-medium text-gray-700">{qa.q}</p>
                                <p className="text-xs text-gray-500 mt-1">{qa.a}</p>
                              </button>
                            </div>
                          ))}
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                ))}
              </div>
            </div>

            {/* Contact Info */}
            <div className="bg-gradient-to-br from-blue-600 to-indigo-700 rounded-xl shadow-lg p-6 mt-6 text-white">
              <h3 className="font-bold mb-4">Need More Help?</h3>
              <div className="space-y-3 text-sm">
                <div className="flex items-center space-x-3">
                  <Phone className="h-4 w-4" />
                  <span>1-800-DEULSOOM</span>
                </div>
                <div className="flex items-center space-x-3">
                  <Mail className="h-4 w-4" />
                  <span>support@deulsoom.com</span>
                </div>
                <div className="flex items-center space-x-3">
                  <Clock className="h-4 w-4" />
                  <span>24/7 Customer Support</span>
                </div>
              </div>
              
              <button
                onClick={() => {
                  if (sessionId) {
                    fetch('/api/v1/customer-service/escalate', {
                      method: 'POST',
                      headers: { 'Content-Type': 'application/json' },
                      body: JSON.stringify({ session_id: sessionId })
                    });
                  }
                  alert('Connecting you to a human agent...');
                }}
                className="w-full mt-4 py-2 bg-white text-blue-600 rounded-lg font-semibold hover:bg-gray-100 transition-colors"
              >
                Talk to Human Agent
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
