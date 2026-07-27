import { useState } from 'react';
import { ConfigProvider, Button, Select, Card, Alert } from 'antd';
import MainLayout from './layouts/MainLayout';
import ImageUploader from './components/ImageUploader';
import PredictionResult from './components/PredictionResult';
import MultiModelResult from './components/MultiModelResult';
import LoadingProgress from './components/LoadingProgress';
import {
  ScanOutlined,
  InfoCircleOutlined,
  ExperimentOutlined
} from '@ant-design/icons';

const NumberIcon = ({ num, colorClass }) => (
  <span className={`inline-flex items-center justify-center w-5 h-5 rounded-full text-[11px] font-black shadow-sm ${colorClass}`}>
    {num}
  </span>
);

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [abResult, setAbResult] = useState(null);
  const [error, setError] = useState(null);
  const [selectedModel, setSelectedModel] = useState("resnet");

  const handleImageSelect = (file, url) => {
    setSelectedFile(file);
    setPreviewUrl(url);
    setResult(null);
    setAbResult(null);
    setError(null);
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    setError(null);
    setResult(null);
    setAbResult(null);

    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("model_name", selectedModel);

    try {
      const response = await fetch("http://localhost:8000/api/v1/ai/predict", {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        throw new Error(`Lỗi server: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      console.error(err);
      setError("Không thể kết nối tới Backend. Hãy chắc chắn Server đang chạy.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleAnalyzeAll = async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    setError(null);
    setResult(null);
    setAbResult(null);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch("http://localhost:8000/api/v1/ai/predict-all", {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        throw new Error(`Lỗi server: ${response.status}`);
      }

      const data = await response.json();
      setAbResult(data);
    } catch (err) {
      console.error(err);
      setError("Không thể kết nối tới Backend. Hãy chắc chắn Server đang chạy tại cổng 8000.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <ConfigProvider
      theme={{
        token: {
          colorPrimary: '#2563eb',
          borderRadius: 12,
          colorLink: '#2563eb',
          fontFamily: 'inherit',
          controlHeight: 40,
        }
      }}
    >
      <MainLayout>

        {/* PAGE INTRO */}
        <div className="mb-8 max-w-2xl">
          <h2 className="text-3xl font-black text-slate-800 tracking-tight leading-tight">Hệ Thống Kiểm Tra Ảnh Giả Mạo</h2>
          <p className="text-slate-500 text-sm mt-2 leading-relaxed">
            Sử dụng kiến trúc mạng nơ-ron học sâu để phân loại ảnh tự nhiên (Real) và ảnh ngụy tạo do Trí tuệ nhân tạo sinh ra (AI-generated).
          </p>
        </div>

        {/* WORKSPACE PANELS */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 lg:gap-8 items-start">

          {/* Controls Card */}
          <div className="lg:col-span-4 flex flex-col gap-6">

            <Card className="rounded-2xl border-slate-200/60 soft-shadow">

              {/* Model select & Uploader block */}
              <div className="flex flex-col gap-6">

                {/* Selector */}
                <div className="flex flex-col items-start bg-slate-50 p-4 rounded-xl border border-slate-200/40 gap-3">
                  <div>
                    <h4 className="text-sm font-bold text-slate-700">Mô hình nhận diện (CNN)</h4>
                    <p className="text-[11px] text-slate-500 mt-0.5 leading-snug">Chọn kiến trúc mạng tối ưu cho phân tích.</p>
                  </div>
                  <Select
                    value={selectedModel}
                    onChange={(val) => setSelectedModel(val)}
                    className="w-full md:w-48 font-medium"
                    options={[
                      {
                        value: 'resnet',
                        label: (
                          <span className="flex items-center gap-2">
                            <NumberIcon num="1" colorClass="bg-blue-100 text-blue-600 border border-blue-200" /> ResNet-50
                          </span>
                        )
                      },
                      {
                        value: 'efficientnet',
                        label: (
                          <span className="flex items-center gap-2">
                            <NumberIcon num="2" colorClass="bg-amber-100 text-amber-600 border border-amber-200" /> EfficientNet-B0
                          </span>
                        )
                      },
                      {
                        value: 'densenet',
                        label: (
                          <span className="flex items-center gap-2">
                            <NumberIcon num="3" colorClass="bg-rose-100 text-rose-600 border border-rose-200" /> DenseNet-121
                          </span>
                        )
                      },
                    ]}
                  />
                </div>

                {/* Uploader */}
                <ImageUploader onImageSelect={handleImageSelect} />

                {/* Submit button */}
                <div className="flex flex-col xl:flex-row justify-center gap-3 mt-2">
                  <Button
                    type="primary"
                    size="large"
                    disabled={!selectedFile || isAnalyzing}
                    loading={isAnalyzing && !abResult && !result}
                    onClick={handleAnalyze}
                    icon={<ScanOutlined />}
                    className="flex-1"
                  >
                    Kiểm tra
                  </Button>
                  <Button
                    type="primary"
                    size="large"
                    disabled={!selectedFile || isAnalyzing}
                    loading={isAnalyzing && !abResult && !result}
                    onClick={handleAnalyzeAll}
                    icon={<ExperimentOutlined />}
                    className="flex-1 font-bold bg-amber-500 hover:!bg-amber-400 border-none shadow-md"
                  >
                    So sánh Mô hình
                  </Button>
                </div>

              </div>

              {error && (
                <div className="mt-6">
                  <Alert
                    message="Lỗi Kết Nối Server"
                    description={error}
                    type="error"
                    showIcon
                    icon={<InfoCircleOutlined />}
                    className="rounded-xl"
                  />
                </div>
              )}
            </Card>
          </div>

          {/* Results Block */}
          <div className="lg:col-span-8">
            {isAnalyzing ? (
              <LoadingProgress />
            ) : abResult ? (
              <MultiModelResult abResult={abResult} originalImgUrl={previewUrl} />
            ) : result ? (
              <PredictionResult result={result} originalImgUrl={previewUrl} />
            ) : (
              <div className="h-full min-h-[400px] border-2 border-dashed border-slate-200 rounded-3xl flex flex-col items-center justify-center text-slate-400 bg-slate-50/50 p-8 text-center transition-all">
                <div className="w-16 h-16 bg-white rounded-full flex items-center justify-center shadow-[0_4px_20px_rgba(0,0,0,0.03)] border border-slate-100 mb-6">
                  <ScanOutlined className="text-2xl text-blue-500/50" />
                </div>
                <h3 className="font-bold text-slate-600 text-base mb-2">Khu vực hiển thị kết quả phân tích</h3>
                <p className="text-xs text-slate-400 max-w-xs leading-relaxed">
                  Tải một bức ảnh lên và bấm "Kiểm tra" hoặc "So sánh Mô hình".
                  Hệ thống AI sẽ quét và trích xuất đặc điểm để đánh giá bức ảnh là Ảnh thật hay do AI tạo sinh, đồng thời trích xuất bản đồ nhiệt (Grad-CAM) để giải thích trực quan.
                </p>
              </div>
            )}
          </div>

        </div>

      </MainLayout>
    </ConfigProvider>
  );
}

export default App;
