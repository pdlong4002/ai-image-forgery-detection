import { 
  WarningOutlined, 
  CheckCircleOutlined, 
  QuestionCircleOutlined,
  ExperimentOutlined,
  FileImageOutlined
} from '@ant-design/icons';
import { Card, Tag, Image } from 'antd';

export default function MultiModelResult({ abResult, originalImgUrl }) {
  if (!abResult || !abResult.success) return null;

  const models = ["resnet", "efficientnet", "densenet"];
  const modelLabels = {
    "resnet": "ResNet-50",
    "efficientnet": "EfficientNet-B0",
    "densenet": "DenseNet-121"
  };

  const getCardStyle = (isFake) => {
    return isFake 
        ? "border-red-200 bg-red-50/20"
        : "border-emerald-200 bg-emerald-50/20";
  };

  return (
    <div className="animate-slide-up flex flex-col gap-6">
      
      {/* Title */}
      <div className="p-6 rounded-2xl flex flex-col items-center justify-center text-center border bg-blue-50/50 border-blue-200 text-blue-800 shadow-sm shadow-blue-100/50 transition-all duration-300">
        <div className="flex items-center gap-3 mb-2">
            <ExperimentOutlined className="text-blue-500" style={{ fontSize: '26px' }} /> 
            <h2 className="text-2xl md:text-3xl font-black tracking-wider leading-none uppercase">
                KẾT QUẢ SO SÁNH CÁC MÔ HÌNH
            </h2>
        </div>
        <p className="text-sm font-semibold text-slate-500">So sánh song song 3 mô hình Deep Learning</p>
      </div>

      {/* Cards Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {models.map(modelKey => {
            const result = abResult.results[modelKey];
            
            if (result.error) {
                return (
                    <Card key={modelKey} className="rounded-2xl border-slate-200 soft-shadow flex flex-col items-center justify-center text-center p-6 bg-slate-50/50">
                        <WarningOutlined className="text-3xl text-orange-400 mb-2" />
                        <h3 className="font-bold text-slate-700">{modelLabels[modelKey]}</h3>
                        <p className="text-xs text-red-500 mt-2">{result.error}</p>
                    </Card>
                );
            }

            const isFake = result.label === "FAKE";
            const confidencePercent = (result.confidence * 100).toFixed(1);
            const fakePercent = (result.fake_probability * 100).toFixed(1);
            const realPercent = (result.real_probability * 100).toFixed(1);

            return (
                <Card 
                    key={modelKey} 
                    className={`rounded-2xl soft-shadow transition-all hover:shadow-lg ${getCardStyle(isFake)}`}
                    bodyStyle={{ padding: '16px' }}
                >
                    {/* Header */}
                    <div className="flex justify-between items-start mb-4">
                        <div>
                            <h3 className="font-extrabold text-slate-800 text-base">{modelLabels[modelKey]}</h3>
                            <div className="text-[10px] text-slate-500 font-medium">Thời gian: {result.processing_time}s</div>
                        </div>
                        <Tag color={isFake ? "error" : "success"} className="rounded-full font-bold m-0 border-0">
                            {result.label}
                        </Tag>
                    </div>

                    {/* Verdict Mini */}
                    <div className="flex items-center gap-2 mb-4">
                        {isFake ? <WarningOutlined className="text-red-500 text-xl" /> : <CheckCircleOutlined className="text-emerald-500 text-xl" />}
                        <span className="font-bold text-sm text-slate-700">Độ tin cậy: {confidencePercent}%</span>
                    </div>

                    {/* Mini Progress Bar */}
                    <div className="w-full h-2 rounded-full flex overflow-hidden bg-slate-100 mb-1">
                        <div className="bg-red-500 h-full progress-transition" style={{ width: `${fakePercent}%` }}></div>
                        <div className="bg-emerald-500 h-full progress-transition" style={{ width: `${realPercent}%` }}></div>
                    </div>
                    <div className="flex justify-between text-[9px] font-bold text-slate-400 mb-4">
                        <span>FAKE {fakePercent}%</span>
                        <span>{realPercent}% REAL</span>
                    </div>

                    {/* Heatmap */}
                    <div className="w-full aspect-square rounded-xl overflow-hidden border border-slate-200/50 bg-slate-100 flex items-center justify-center p-1">
                        {result.heatmap_base64 ? (
                            <Image 
                                src={result.heatmap_base64} 
                                alt={`Heatmap ${modelKey}`} 
                                rootClassName="w-full h-full"
                                className="w-full h-full object-cover rounded-lg"
                                style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                            />
                        ) : (
                            <div className="flex flex-col items-center text-slate-300">
                                <QuestionCircleOutlined className="text-2xl mb-1" />
                                <span className="text-[10px]">No Grad-CAM</span>
                            </div>
                        )}
                    </div>
                </Card>
            );
        })}
      </div>

      {/* Source Image reference */}
      <Card className="rounded-2xl border-slate-200/60 soft-shadow">
          <div className="flex flex-col md:flex-row items-center gap-6">
             <div className="w-24 h-24 rounded-lg overflow-hidden border border-slate-200 shrink-0 flex items-center justify-center bg-slate-50">
                <Image 
                    src={originalImgUrl} 
                    alt="Original" 
                    rootClassName="w-full h-full"
                    className="w-full h-full object-cover" 
                    style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                />
             </div>
             <div>
                <h4 className="font-bold text-slate-700 flex items-center gap-2">
                    <FileImageOutlined /> Ảnh gốc phân tích
                </h4>
                <p className="text-xs text-slate-500 mt-1 max-w-lg leading-relaxed">
                    Hệ thống chạy cùng một bức ảnh gốc qua cả 3 kiến trúc mạng nơ-ron khác nhau để trích xuất các đặc trưng bất thường, giúp phân loại ảnh AI tạo sinh chính xác và chống nhiễu loạn mô hình.
                </p>
             </div>
          </div>
      </Card>
    </div>
  );
}
