import React, { useState, useRef } from 'react';
import FlowCanvas from './FlowCanvas';
import UploadLink from './UploadLink';
import FloatingPanel from './FloatingPanel';
import { ReactFlowProvider } from '@xyflow/react';
import './index.css';


interface Flow {
  id: string;
  jsonPath: string; // Each flow now has its own jsonPath
  // Add other properties as needed
}

const App: React.FC = () => {
  const [selectedFlowId, setSelectedFlowId] = useState<string | null>(null);
  const [flows, setFlows] = useState<Flow[]>([]); // Array of flows with jsonPath
  const containerRef = useRef<HTMLDivElement>(null);

  // Function to handle the JSON path update for a specific flow
  const handleUploadSuccess = (jsonPath: string, flowId: string) => {
    setFlows((prevFlows) =>
      prevFlows.map((flow) =>
        flow.id === flowId ? { ...flow, jsonPath } : flow
      )
    );
    console.log("Updated jsonPath for flow:", flowId, jsonPath);
  };

  // Toggle canvas width
  const toggleFlowCanvasWidth = (id: string) => {
    setSelectedFlowId(id === selectedFlowId ? null : id);
  };

  const handleDeleteFlow = (id: string) => {
    setFlows(flows.filter(flow => flow.id !== id));
  };

  return (
    <div className="app-container">
      {/* upload */}
      <div className="floating-panel-container">
        <FloatingPanel>
          {/* Pass the handleUploadSuccess function and flow id to UploadLink */}
          {flows.map(flow => (
            <UploadLink
              key={flow.id}
              onUploadSuccess={(jsonPath) => handleUploadSuccess(jsonPath, flow.id)}
            />
          ))}
        </FloatingPanel>
      </div>

      {/* flows */}
      <div ref={containerRef} className={`flows-container ${selectedFlowId ? 'centered' : ''}`}>
        {flows
          .filter((flow) => !selectedFlowId || flow.id === selectedFlowId)
          .map((flow) => (
            <ReactFlowProvider key={flow.id}>
              <FlowCanvas
                jsonPath={flow.jsonPath} // Use the individual jsonPath for each flow
                canvasWidth={selectedFlowId === flow.id || flows.length === 1 ? 1200 : 800}
                rootWidth={600}
                onClick={() => toggleFlowCanvasWidth(flow.id)} 
                onDelete={() => handleDeleteFlow(flow.id)}
              />
            </ReactFlowProvider>
          ))}
      </div>
    </div>
  );
};

export default App;
