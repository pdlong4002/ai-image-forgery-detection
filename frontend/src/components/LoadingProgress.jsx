import { LoadingOutlined, ScanOutlined } from '@ant-design/icons';

export default function LoadingProgress() {
  return (
    <div className="h-full min-h-[400px] border border-blue-100 rounded-3xl flex flex-col items-center justify-center text-slate-600 bg-blue-50/30 p-8 text-center relative overflow-hidden shadow-inner">
        {/* Animated background pulse */}
        <div className="absolute inset-0 bg-gradient-to-tr from-blue-100/40 to-indigo-100/40 animate-pulse"></div>

        <div className="relative z-10 flex flex-col items-center w-full">
            <div className="relative mb-8">
                <div className="text-blue-500 text-6xl opacity-30 absolute inset-0 flex items-center justify-center animate-ping">
                    <ScanOutlined />
                </div>
                <div className="text-blue-600 text-6xl relative z-10">
                    <ScanOutlined />
                </div>
            </div>
            
            <h3 className="font-extrabold text-slate-800 text-xl md:text-2xl mb-3 tracking-wider uppercase flex items-center gap-3">
                <LoadingOutlined className="text-blue-500" />
                AI Đang Phân Tích Hình Ảnh
            </h3>
            
            <p className="text-slate-500 font-medium text-sm max-w-sm">
                Hệ thống đang quét đặc trưng và đánh giá ảnh. Quá trình này có thể mất vài giây, vui lòng đợi...
            </p>
        </div>
    </div>
  );
}
