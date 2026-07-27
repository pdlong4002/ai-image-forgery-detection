import { useState, useRef } from 'react';
import { Upload, Image as ImageIcon } from 'lucide-react';
import { Button, message } from 'antd';

export default function ImageUploader({ onImageSelect }) {
  const [dragActive, setDragActive] = useState(false);
  const [preview, setPreview] = useState(null);
  const inputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (file) => {
    if (!file.type.startsWith('image/')) {
      message.error('Vui lòng chọn một file hình ảnh (JPG, PNG)');
      return;
    }
    
    const MAX_SIZE = 10 * 1024 * 1024; // 10MB
    if (file.size > MAX_SIZE) {
      message.error('Kích thước ảnh vượt quá giới hạn 10MB. Vui lòng chọn ảnh có dung lượng nhỏ hơn.');
      return;
    }

    const url = URL.createObjectURL(file);
    setPreview(url);
    onImageSelect(file, url);
  };

  const onButtonClick = () => {
    inputRef.current.click();
  };

  return (
    <div 
      className={`relative rounded-2xl border-2 border-dashed p-6 flex flex-col items-center justify-center text-center transition-all duration-300 cursor-pointer
        ${dragActive ? 'border-blue-500 bg-blue-50/50' : 'border-slate-200 bg-slate-50/30 hover:border-blue-500 hover:bg-slate-50/80'} 
        ${preview ? 'py-4' : 'py-10'}`}
      onDragEnter={handleDrag}
      onDragLeave={handleDrag}
      onDragOver={handleDrag}
      onDrop={handleDrop}
      onClick={onButtonClick}
    >
      <input
        ref={inputRef}
        type="file"
        className="hidden"
        accept="image/*"
        onChange={handleChange}
      />
      
      {preview ? (
        <div className="flex flex-col items-center gap-4">
            <div className="relative group">
              <img src={preview} alt="Preview" className="h-52 w-auto object-contain rounded-xl border border-slate-200 shadow-md bg-white p-1" />
              <div className="absolute inset-0 bg-black/40 rounded-xl opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center text-white text-xs font-semibold">
                Thay đổi ảnh
              </div>
            </div>
            <p className="text-xs text-slate-400 font-medium">Bấm vào ảnh hoặc kéo thả để đổi ảnh khác</p>
        </div>
      ) : (
        <>
            <div className="w-16 h-16 bg-blue-50 rounded-full flex items-center justify-center mb-4 text-blue-600 border border-blue-100 shadow-sm">
                <Upload size={28} />
            </div>
            <h3 className="text-lg font-bold text-slate-800 mb-1">Tải ảnh lên để phân tích</h3>
            <p className="text-slate-500 text-sm mb-1">Kéo thả file vào đây, hoặc nhấn để chọn</p>
            <p className="text-slate-400 text-[11px] mt-2 font-medium bg-slate-100/50 px-2 py-1 rounded-md">Hỗ trợ định dạng JPG, PNG (Tối đa 10MB)</p>
        </>
      )}
    </div>
  );
}
