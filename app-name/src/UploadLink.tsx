import React, { useState } from 'react';

// Update interface to accept jsonPath in the onUploadSuccess callback
interface UploadLinkProps {
  onUploadSuccess: (jsonPath: string) => void; // Define jsonPath parameter
}

// Use React.FC<UploadLinkProps> to specify component prop types
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
      const jsonPath = data.jsonPath; // Store jsonPath
      setResponse(data.link ? `Received link: ${data.link}` : data.message);

      // Call onUploadSuccess with jsonPath after successful upload
      onUploadSuccess(jsonPath);

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
