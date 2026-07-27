import React, { useState, useEffect } from 'react';
import { 
  GithubOutlined,
  SyncOutlined
} from '@ant-design/icons';
import { Avatar, Tooltip } from 'antd';
import { ScanSearch } from 'lucide-react';

export default function MainLayout({ children }) {
  const [isOnline, setIsOnline] = useState(false);

  useEffect(() => {
    const checkServerStatus = async () => {
      try {
        // Ping Backend
        const res = await fetch("http://localhost:8000/");
        setIsOnline(res.ok);
      } catch (err) {
        setIsOnline(false);
      }
    };

    // Gọi lần đầu
    checkServerStatus();

    // Check lại mỗi 5 giây
    const intervalId = setInterval(checkServerStatus, 5000);
    return () => clearInterval(intervalId);
  }, []);

  return (
    <div className="flex flex-col min-h-screen bg-slate-50 font-sans text-slate-700">
      
      {/* HEADER */}
      <header className="h-[72px] bg-white border-b border-slate-200/80 px-8 flex items-center justify-between shrink-0 z-10 shadow-[0_2px_10px_rgba(0,0,0,0.01)]">
        {/* LOGO */}
        <div className="flex items-center gap-3 cursor-pointer hover:scale-[1.01] transition-transform">
          <div className="w-10 h-10 shrink-0 bg-gradient-to-tr from-blue-600 to-indigo-600 rounded-[12px] flex items-center justify-center text-white shadow-md shadow-blue-500/20">
            <ScanSearch size={22} strokeWidth={2.5} />
          </div>
          <div className="flex flex-col">
            <h1 className="text-lg font-black text-slate-800 tracking-tight leading-none">ForgeryScan</h1>
            <span className="text-[10px] font-bold text-blue-600 uppercase tracking-widest mt-1">Deep Learning XAI</span>
          </div>
        </div>

        {/* RIGHT SIDE GROUP */}
        <div className="flex items-center gap-6">
          {/* API STATUS */}
          {isOnline ? (
            <div className="hidden md:flex items-center gap-2 bg-emerald-50 text-emerald-700 px-3 py-1.5 rounded-full text-xs font-semibold border border-emerald-100 transition-colors duration-500">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
              FastAPI Online
            </div>
          ) : (
            <div className="hidden md:flex items-center gap-2 bg-rose-50 text-rose-700 px-3 py-1.5 rounded-full text-xs font-semibold border border-rose-100 transition-colors duration-500">
              <span className="w-2 h-2 rounded-full bg-rose-500"></span>
              FastAPI Offline
            </div>
          )}

          {/* PROFILE/RIGHT BAR */}
          <div className="flex items-center gap-4">
            <Tooltip title="Github Project">
              <a 
                href="https://github.com/pdlong4002/ai-image-forgery-detection" 
                target="_blank" 
                rel="noreferrer"
                className="p-2 hover:bg-slate-100 rounded-full transition-colors text-slate-500 hover:text-slate-800"
              >
                <GithubOutlined style={{ fontSize: '18px' }} />
              </a>
            </Tooltip>
          </div>
        </div>
      </header>

      {/* MAIN CONTAINER */}
      <main className="flex-grow py-10 px-4 md:px-8">
        <div className="max-w-[1500px] mx-auto w-full">
          {children}
        </div>
      </main>

      {/* FOOTER */}
      <footer className="bg-white border-t border-slate-200 py-6 px-8 flex flex-col md:flex-row items-center justify-between text-xs text-slate-400 gap-4 shrink-0 shadow-[0_-2px_10px_rgba(0,0,0,0.01)]">
        <div>
          &copy; 2026 <strong className="text-slate-600 font-bold">ForgeryScan Pro</strong>. Khóa luận Tốt nghiệp - Ứng dụng Học sâu (Deep Learning) trong Giám định Hình ảnh.
        </div>
        <div className="flex items-center gap-4">
          <span><strong>AI Core:</strong> PyTorch (Grad-CAM)</span>
          <span>•</span>
          <span><strong>Backend:</strong> FastAPI</span>
          <span>•</span>
          <span><strong>Frontend:</strong> React (Ant Design)</span>
        </div>
      </footer>

    </div>
  );
}
