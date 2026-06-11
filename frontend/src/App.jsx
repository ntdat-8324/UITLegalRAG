import React, { useState, useEffect, useRef } from 'react';
import { Send, Menu, Plus, MessageSquare, Bookmark, Trash2, Shield, Copy, Check } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import axios from 'axios';

const API_BASE = 'http://localhost:8000/api';

function App() {
  const [sessions, setSessions] = useState([]);
  const [currentSessionId, setCurrentSessionId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [bookmarks, setBookmarks] = useState([]);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const messagesEndRef = useRef(null);

  // Load from local storage
  useEffect(() => {
    const savedSessions = JSON.parse(localStorage.getItem('legal_sessions')) || [];
    const savedBookmarks = JSON.parse(localStorage.getItem('legal_bookmarks')) || [];
    setSessions(savedSessions);
    setBookmarks(savedBookmarks);
    
    if (savedSessions.length > 0) {
      loadSession(savedSessions[0].id);
    } else {
      createNewSession();
    }
  }, []);

  // Save to local storage on change
  useEffect(() => {
    if (sessions.length > 0) {
      localStorage.setItem('legal_sessions', JSON.stringify(sessions));
    }
  }, [sessions]);

  useEffect(() => {
    localStorage.setItem('legal_bookmarks', JSON.stringify(bookmarks));
  }, [bookmarks]);

  // Auto-scroll
  useEffect(() => {
    // Disable smooth scrolling during fast streaming updates to avoid UI locking/jerkiness
    messagesEndRef.current?.scrollIntoView({ behavior: isLoading ? 'auto' : 'smooth' });
  }, [messages, isLoading]);

  const createNewSession = () => {
    const newSession = {
      id: Date.now().toString(),
      title: 'New Chat',
      messages: []
    };
    setSessions([newSession, ...sessions]);
    setCurrentSessionId(newSession.id);
    setMessages([]);
  };

  const loadSession = (id) => {
    setCurrentSessionId(id);
    const session = sessions.find(s => s.id === id);
    if (session) {
      setMessages(session.messages);
    }
  };

  const updateSessionMessages = (newMessages) => {
    setMessages(newMessages);
    setSessions(prev => prev.map(s => {
      if (s.id === currentSessionId) {
        // Update title based on first user message if it's "New Chat"
        let title = s.title;
        if (title === 'New Chat' && newMessages.length > 0) {
          title = newMessages[0].content.substring(0, 30) + (newMessages[0].content.length > 30 ? '...' : '');
        }
        return { ...s, title, messages: newMessages };
      }
      return s;
    }));
  };

  const deleteSession = (e, id) => {
    e.stopPropagation();
    const updated = sessions.filter(s => s.id !== id);
    setSessions(updated);
    if (updated.length > 0) {
      if (currentSessionId === id) loadSession(updated[0].id);
    } else {
      createNewSession();
    }
  };

  const toggleBookmark = (messageId, content) => {
    const exists = bookmarks.find(b => b.id === messageId);
    if (exists) {
      setBookmarks(bookmarks.filter(b => b.id !== messageId));
    } else {
      setBookmarks([...bookmarks, { id: messageId, content, date: new Date().toISOString() }]);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMsg = { id: Date.now().toString(), role: 'user', content: input };
    const newMessages = [...messages, userMsg];
    updateSessionMessages(newMessages);
    setInput('');
    setIsLoading(true);

    // Create an empty AI message placeholder
    const aiMsgId = (Date.now() + 1).toString();
    const initialAiMsg = { id: aiMsgId, role: 'assistant', content: '', sources: [] };
    let currentAiMsg = { ...initialAiMsg };
    updateSessionMessages([...newMessages, currentAiMsg]);

    try {
      const response = await fetch(`${API_BASE}/query_stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: input,
          top_k: 5,
          use_classifier: true,
          use_reranker: true
        })
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        buffer += decoder.decode(value, { stream: true });
        
        const lines = buffer.split('\n');
        buffer = lines.pop(); // Keep the last incomplete line in buffer

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const dataStr = line.substring(6);
            if (!dataStr.trim()) continue;
            try {
              const data = JSON.parse(dataStr);
              if (data.type === 'sources') {
                currentAiMsg.sources = data.data;
              } else if (data.type === 'chunk') {
                currentAiMsg.content += data.content;
              }
              // Update state for re-render
              updateSessionMessages([...newMessages, { ...currentAiMsg }]);
            } catch (err) {
              console.error("Error parsing stream data", err, dataStr);
            }
          }
        }
      }
    } catch (error) {
      console.error(error);
      currentAiMsg.content = 'Xin lỗi, đã xảy ra lỗi kết nối với máy chủ. Vui lòng thử lại sau.';
      updateSessionMessages([...newMessages, currentAiMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className="app-container">
      {/* Sidebar */}
      <div className={`sidebar ${isSidebarOpen ? 'open' : ''}`}>
        <div className="sidebar-header">
          <Shield size={24} color="#3b82f6" />
          <span>Legal RAG AI</span>
        </div>
        <button className="new-chat-btn" onClick={createNewSession}>
          <Plus size={18} /> New Chat
        </button>
        
        <div className="sidebar-content">
          <div className="sidebar-section">
            <div className="sidebar-title">Recent Chats</div>
            {sessions.map(s => (
              <div 
                key={s.id} 
                className={`history-item ${s.id === currentSessionId ? 'active' : ''}`}
                onClick={() => loadSession(s.id)}
              >
                <MessageSquare size={16} />
                <span style={{flex: 1}}>{s.title}</span>
                <Trash2 size={14} className="delete-icon" onClick={(e) => deleteSession(e, s.id)} style={{color: '#64748b'}}/>
              </div>
            ))}
          </div>

          {bookmarks.length > 0 && (
            <div className="sidebar-section">
              <div className="sidebar-title">Bookmarks</div>
              {bookmarks.map(b => (
                <div key={b.id} className="history-item">
                  <Bookmark size={16} color="#fbbf24" />
                  <span>{b.content.substring(0, 25)}...</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Main Content */}
      <div className="main-content">
        <div className="chat-area">
          {messages.length === 0 ? (
            <div className="welcome-screen">
              <Shield size={48} color="#3b82f6" style={{marginBottom: '1rem'}}/>
              <h1>Hệ Thống Trợ Lý Pháp Luật</h1>
              <p>Đặt câu hỏi về các quy chế, điều luật của trường Đại học Công nghệ Thông tin (UIT) và các văn bản pháp lý khác.</p>
              
              <div className="suggestions">
                <div className="suggestion-card" onClick={() => setInput('Trường UIT là gì?')}>
                  Trường UIT là gì?
                </div>
                <div className="suggestion-card" onClick={() => setInput('Điều kiện xét học bổng khuyến khích học tập là gì?')}>
                  Điều kiện xét học bổng khuyến khích học tập là gì?
                </div>
                <div className="suggestion-card" onClick={() => setInput('Sinh viên vi phạm quy chế thi sẽ bị xử lý như thế nào?')}>
                  Sinh viên vi phạm quy chế thi sẽ bị xử lý như thế nào?
                </div>
                <div className="suggestion-card" onClick={() => setInput('Quy định về thời gian tối đa hoàn thành khoá học?')}>
                  Quy định về thời gian tối đa hoàn thành khoá học?
                </div>
              </div>
            </div>
          ) : (
            messages.map((msg) => (
              <div key={msg.id} className={`message-container ${msg.role === 'user' ? 'message-user' : 'message-ai'}`}>
                <div className={`avatar ${msg.role === 'user' ? 'avatar-user' : 'avatar-ai'}`}>
                  {msg.role === 'user' ? 'U' : <Shield size={20} color="#fff" />}
                </div>
                  <div className="message-content">
                    <div className="message-text">
                      {msg.role === 'assistant' && !msg.content ? (
                        <div className="typing-indicator" style={{marginTop: '8px'}}>
                          <div className="typing-dot"></div>
                          <div className="typing-dot"></div>
                          <div className="typing-dot"></div>
                        </div>
                      ) : (
                        <ReactMarkdown>{msg.content}</ReactMarkdown>
                      )}
                    </div>
                    
                    {msg.role === 'assistant' && msg.content && (
                      <div className="message-actions">
                        <CopyButton text={msg.content} />
                      <button 
                        className={`action-btn ${bookmarks.find(b=>b.id===msg.id) ? 'active' : ''}`}
                        onClick={() => toggleBookmark(msg.id, msg.content)}
                        title="Bookmark"
                      >
                        <Bookmark size={16} />
                      </button>
                    </div>
                  )}

                  {msg.sources && msg.sources.length > 0 && (
                    <div className="sources-container">
                      <div className="sources-title">
                        <Bookmark size={14} /> Nguồn tham khảo
                      </div>
                      <div className="source-grid">
                        {msg.sources.map((src, i) => (
                          <div key={i} className="source-card">
                            <h4>Điều {src.article}</h4>
                            <div className="doc-name">{src.document}</div>
                            <div className="context">{src.context}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="input-area">
          <form className="input-container" onSubmit={handleSubmit}>
            <textarea
              className="chat-input"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Hỏi về quy chế, văn bản pháp luật..."
              rows={1}
            />
            <button 
              type="submit" 
              className="submit-btn" 
              disabled={!input.trim() || isLoading}
            >
              <Send size={18} />
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}

// Simple Copy Button component
function CopyButton({ text }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy text', err);
    }
  };

  return (
    <button className="action-btn" onClick={handleCopy} title="Copy">
      {copied ? <Check size={16} color="#22c55e" /> : <Copy size={16} />}
    </button>
  );
}

export default App;
