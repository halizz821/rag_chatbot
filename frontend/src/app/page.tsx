"use client";

import React, { useState, useRef, useEffect } from 'react';
import styles from './page.module.css';

type TelemetryStatus = 'pending' | 'partitioning' | 'chunking' | 'summarizing' | 'indexing' | 'complete' | 'error';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  sources?: string[];
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<TelemetryStatus>('pending');
  const [statusMessage, setStatusMessage] = useState<string>('Awaiting document upload...');
  const [messages, setMessages] = useState<Message[]>([]);
  const [query, setQuery] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [isChatting, setIsChatting] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setIsUploading(true);
    setStatus('partitioning');
    setStatusMessage('Initiating ingestion engine...');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/api/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Upload failed');
      if (!response.body) throw new Error('No response body');

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n\n');
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const dataStr = line.substring(6);
            if (dataStr.trim()) {
              try {
                const data = JSON.parse(dataStr);
                setStatus(data.status);
                setStatusMessage(data.message);
              } catch (e) {
                console.error("Parse error", e);
              }
            }
          }
        }
      }
    } catch (err) {
        console.error(err);
        setStatus('error');
        setStatusMessage('An error occurred during ingestion.');
    } finally {
      setIsUploading(false);
    }
  };

  const handleSend = async () => {
    if (!query.trim()) return;
    
    const userMsg: Message = { role: 'user', content: query };
    setMessages(prev => [...prev, userMsg]);
    setQuery('');
    setIsChatting(true);

    try {
      const response = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: userMsg.content }),
      });

      if (!response.ok) throw new Error('Chat failed');
      const data = await response.json();
      
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: data.answer,
        sources: data.sources 
      }]);
    } catch (err) {
      console.error(err);
      setMessages(prev => [...prev, { role: 'assistant', content: "An error occurred while generating a response." }]);
    } finally {
      setIsChatting(false);
    }
  };

  const renderTelemetry = () => {
    const phases = ['partitioning', 'chunking', 'summarizing', 'indexing', 'complete'];
    const activeIndex = phases.indexOf(status);

    return (
      <div className={styles.telemetry}>
        <div className={styles.phases}>
          {phases.map((phase, i) => {
             const isActive = i === activeIndex;
             const isPast = i < activeIndex || status === 'complete';
             return (
               <div key={phase} className={`${styles.phase} ${isActive ? styles.activePhase : ''} ${isPast ? styles.pastPhase : ''}`}>
                 <div className={styles.phaseDot}></div>
                 <span className={styles.phaseLabel}>{phase}</span>
               </div>
             )
          })}
        </div>
        <p className={styles.statusMessage}>{statusMessage}</p>
      </div>
    );
  };

  return (
    <main className={styles.main}>
      <header className={styles.header}>
        <h1>Intelligent Document Assistant</h1>
        <p>High-fidelity, dual-module retrieval architecture.</p>
      </header>

      <div className={styles.contentWrapper}>
        <section className={styles.chatSection}>
          <div className={styles.glassCard}>
            <h2>Hybrid Retrieval Synthesis</h2>
            
            <div className={styles.chatBox}>
              {messages.length === 0 ? (
                <div className={styles.emptyState}>
                  Upload a document and ask a question to engage the Hybrid Retrieval mechanism.
                </div>
              ) : (
                  messages.map((msg, idx) => (
                      <div key={idx} className={`${styles.message} ${msg.role === 'user' ? styles.userMsg : styles.aiMsg}`}>
                         <p>{msg.content}</p>
                         {msg.sources && msg.sources.length > 0 && (
                            <div className={styles.sourcesWrapper}>
                               <span className={styles.sourceLabel}>Sources: </span>
                               {[...new Set(msg.sources)].map(src => (
                                   <span key={src as string} className={styles.sourceTag}>{src}</span>
                               ))}
                            </div>
                         )}
                      </div>
                  ))
              )}
              {isChatting && (
                  <div className={`${styles.message} ${styles.aiMsg} ${styles.loadingMsg}`}>
                     <span className={styles.dotPulse}></span>
                  </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            <div className={styles.inputArea}>
              <input 
                type="text" 
                value={query}
                onChange={e => setQuery(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && handleSend()}
                placeholder="Query the system..."
                className={styles.chatInput}
                disabled={isChatting}
              />
              <button 
                onClick={handleSend} 
                className={styles.sendButton}
                disabled={isChatting || !query.trim()}
              >
                <svg className={styles.icon} fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" /></svg>
                Send
              </button>
            </div>
          </div>
        </section>

        <section className={styles.uploadSection}>
          <div className={styles.glassCard}>
            <h2>Ingestion Engine</h2>
            <div className={styles.uploadControls}>
              <input 
                type="file" 
                accept=".pdf" 
                onChange={handleFileChange} 
                className={styles.fileInput}
                id="file-upload"
              />
              <label htmlFor="file-upload" className={styles.fileLabel}>
                <svg className={styles.icon} fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 13h6m-3-3v6m5 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" /></svg>
                {file ? file.name : "Select PDF Document"}
              </label>
              <button 
                onClick={handleUpload} 
                disabled={!file || isUploading}
                className={styles.primaryButton}
              >
                <svg className={styles.icon} fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" /></svg>
                {isUploading ? "Processing..." : "Commence Ingestion"}
              </button>
            </div>
            
            {status !== 'pending' && renderTelemetry()}
          </div>
        </section>
      </div>
    </main>
  );
}
