import React, { useState } from 'react';

// 添加接口，定义组件的属性类型
interface UploadLinkProps {
  onUploadSuccess: () => void; // 定义 onUploadSuccess 回调的类型
}

// 使用 React.FC<UploadLinkProps> 来指定组件的属性类型
const UploadLink: React.FC<UploadLinkProps> = ({ onUploadSuccess }) => {
  const [link, setLink] = useState('');
  const [response, setResponse] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    try {
      const res = await fetch('http://127.0.0.1:8000/api/upload-link/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ link }),
      });

      const data = await res.json();
      setResponse(data.link ? `Received link: ${data.link}` : data.message);

      // 上传成功后调用 onUploadSuccess
      onUploadSuccess();

    } catch (error) {
      console.error("Error:", error);
      setResponse("An error occurred while processing your link.");
    }
  };

  return (
    <div>
      <h2>Upload Link</h2>
      <form onSubmit={handleSubmit}>
        <input 
          type="text" 
          placeholder="Enter link" 
          value={link} 
          onChange={(e) => setLink(e.target.value)} 
          required 
        />
        <button type="submit">Submit</button>
      </form>
      {response && <p>Response: {response}</p>}
    </div>
  );
};

export default UploadLink;
