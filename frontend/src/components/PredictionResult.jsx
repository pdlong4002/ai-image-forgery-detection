import { 
  RobotOutlined, 
  ExpandOutlined, 
  FieldTimeOutlined, 
  WarningOutlined, 
  CheckCircleOutlined, 
  QuestionCircleOutlined 
} from '@ant-design/icons';
import { Card, Image } from 'antd';

export default function PredictionResult({ result, originalImgUrl }) {
  if (!result) return null;

  const isFake = result.label === "FAKE";
  const confidencePercent = (result.confidence * 100).toFixed(1);
  const fakePercent = (result.fake_probability * 100).toFixed(1);
  const realPercent = (result.real_probability * 100).toFixed(1);

  return (
    <div className="animate-slide-up flex flex-col gap-6">
      
      {/* VERDICT BANNER */}
      <div 
        className={`p-6 rounded-2xl flex flex-col items-center justify-center text-center border transition-all duration-300
          ${isFake 
            ? 'bg-red-50/50 border-red-200 text-red-700 shadow-sm shadow-red-100/50' 
            : 'bg-emerald-50/50 border-emerald-200 text-emerald-700 shadow-sm shadow-emerald-100/50'
          }`}
      >
        <div className="flex items-center gap-3 mb-2">
            {isFake ? <WarningOutlined className="text-red-500 animate-bounce" style={{ fontSize: '26px' }} /> : <CheckCircleOutlined className="text-emerald-500 animate-pulse" style={{ fontSize: '26px' }} />}
            <h2 className="text-3xl font-black tracking-wider leading-none">
              ẢNH CÓ THỂ LÀ {result.label}
            </h2>
        </div>
        <p className="text-sm font-semibold text-slate-500">
          Độ tin cậy của thuật toán: <span className="font-extrabold text-slate-800 text-base">{confidencePercent}%</span>
        </p>
      </div>

      {/* HORIZONTAL BAR CHART */}
      <Card className="rounded-2xl border-slate-200/60 soft-shadow">
        <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-4 text-center">
          Tỉ Lệ Đối Kháng Xác Suất (Probability Bar)
        </h3>
        
        <div className="flex justify-between text-xs font-bold mb-2">
            <span className="text-red-600">FAKE {fakePercent}%</span>
            <span className="text-emerald-600">{realPercent}% REAL</span>
        </div>
        
        {/* Progress Bar Container */}
        <div className="w-full h-5 rounded-full flex overflow-hidden bg-slate-100 border border-slate-200/40">
            <div 
                className="bg-red-500 h-full progress-transition" 
                style={{ width: `${fakePercent}%` }}
            ></div>
            <div 
                className="bg-emerald-500 h-full progress-transition" 
                style={{ width: `${realPercent}%` }}
            ></div>
        </div>
        
        <div className="mt-3 flex justify-between text-[10px] text-slate-400 font-medium">
          <span>Khuyên dùng mô hình EfficientNet để có độ chuẩn xác tối ưu nhất</span>
          <span>Phương pháp XAI: Grad-CAM</span>
        </div>
      </Card>

      {/* STATS ROW */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 xl:gap-4">
        {/* Model info */}
        <div className="bg-white border border-slate-200/60 rounded-2xl p-3 xl:p-4 flex items-center gap-3 soft-shadow transition-all hover:-translate-y-1 hover:shadow-md min-w-0">
            <div className="w-10 h-10 xl:w-12 xl:h-12 rounded-full flex items-center justify-center text-lg xl:text-xl bg-blue-50 text-blue-500 shrink-0">
              <RobotOutlined />
            </div>
            <div className="min-w-0 flex-1">
                <span className="text-[11px] xl:text-xs text-slate-500 font-semibold block truncate">Thuật toán AI</span>
                <p className="text-base xl:text-lg font-extrabold text-slate-900 leading-none mt-1 truncate uppercase">{result.model_name || 'ResNet50'}</p>
            </div>
        </div>

        {/* Size info */}
        <div className="bg-white border border-slate-200/60 rounded-2xl p-3 xl:p-4 flex items-center gap-3 soft-shadow transition-all hover:-translate-y-1 hover:shadow-md min-w-0">
            <div className="w-10 h-10 xl:w-12 xl:h-12 rounded-full flex items-center justify-center text-lg xl:text-xl bg-purple-50 text-purple-500 shrink-0">
              <ExpandOutlined />
            </div>
            <div className="min-w-0 flex-1">
                <span className="text-[11px] xl:text-xs text-slate-500 font-semibold block truncate">Kích thước ảnh</span>
                <p className="text-base xl:text-lg font-extrabold text-slate-900 leading-none mt-1 truncate">{result.image_size || 'N/A'}</p>
            </div>
        </div>

        {/* Time info */}
        <div className="bg-white border border-slate-200/60 rounded-2xl p-3 xl:p-4 flex items-center gap-3 soft-shadow transition-all hover:-translate-y-1 hover:shadow-md min-w-0">
            <div className="w-10 h-10 xl:w-12 xl:h-12 rounded-full flex items-center justify-center text-lg xl:text-xl bg-orange-50 text-orange-500 shrink-0">
              <FieldTimeOutlined />
            </div>
            <div className="min-w-0 flex-1">
                <span className="text-[11px] xl:text-xs text-slate-500 font-semibold block truncate">Thời gian xử lý</span>
                <p className="text-base xl:text-lg font-extrabold text-slate-900 leading-none mt-1 truncate">{result.processing_time || '0.0'}s</p>
            </div>
        </div>
      </div>

      {/* IMAGES COMPARISON */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-2">
        <Card title="Ảnh Gốc tải lên" className="rounded-2xl border-slate-200/60 soft-shadow" headStyle={{ fontWeight: 700 }}>
            <div className="w-full rounded-xl overflow-hidden border border-slate-100 bg-slate-50 flex items-center justify-center p-2">
                <Image 
                    src={originalImgUrl} 
                    alt="Original" 
                    rootClassName="w-full"
                    className="w-full h-auto object-contain rounded-lg shadow-sm" 
                />
            </div>
        </Card>
        
        <Card 
            title={isFake ? "Vùng nghi ngờ giả mạo (XAI Grad-CAM)" : "Vùng tự nhiên đáng tin cậy (XAI Grad-CAM)"} 
            className="rounded-2xl border-slate-200/60 soft-shadow" 
            headStyle={{ fontWeight: 700 }}
        >
            <div className="w-full rounded-xl overflow-hidden border border-slate-100 bg-slate-50 flex items-center justify-center p-2">
                {result.heatmap_base64 ? (
                    <Image 
                        src={result.heatmap_base64} 
                        alt="Heatmap" 
                        rootClassName="w-full"
                        className="w-full h-auto object-contain rounded-lg shadow-sm" 
                    />
                ) : (
                    <div className="flex flex-col items-center text-slate-400 text-center p-4">
                        <QuestionCircleOutlined className="mb-2 opacity-55 animate-pulse" style={{ fontSize: '32px' }} />
                        <p className="font-semibold text-sm">Grad-CAM Không hoạt động</p>
                        <p className="text-xs max-w-xs mt-1">Mô hình chưa tải thành công trọng số nên không thể phân tích vùng ảnh giả mạo.</p>
                    </div>
                )}
            </div>
        </Card>
      </div>

    </div>
  );
}
