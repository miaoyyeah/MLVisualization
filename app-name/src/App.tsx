import { useState, useRef } from "react";
import FlowCanvas from "./FlowCanvasNew";
import UploadLink from "./UploadLink";
import FloatingPanel from "./FloatingPanel";
import { ReactFlowProvider } from "@xyflow/react";
import "./index.css";

export default function App() {
  const [flows, setFlows] = useState<
    { id: number; json_path:string ,nodes: any[]; edges: any[] }[]
  >([]);
  const [selectedFlowId, setSelectedFlowId] = useState<number | null>(null);

  const containerRef = useRef<HTMLDivElement>(null);

  const handleUploadSuccess = (json_path:string) => {
    const newFlow = {
      id: Date.now(),
      nodes: [
        {
          id: "1",
          type: "input",
          data: { label: "New Flow Node" },
          position: { x: 250, y: 5 },
        },
      ],
      edges: [],
      json_path: json_path
    };

    setFlows((prevFlows) => [...prevFlows, newFlow]);
  };

  if (containerRef.current) {
    containerRef.current.scrollLeft = containerRef.current.scrollWidth;
  }

  // switch flow canvas width
  const toggleFlowCanvasWidth = (id: number) => {
    setSelectedFlowId((prevId) => (prevId === id ? null : id));
  };

  const handleDeleteFlow = (id: number) => {
    setFlows((prevFlows) => prevFlows.filter((flow) => flow.id !== id));
    if (selectedFlowId === id) {
      setSelectedFlowId(null);
    }
  };

  return (
    <div className="app-container">
      {/* upload */}
      <div className="floating-panel-container">
        <FloatingPanel>
          <UploadLink onUploadSuccess={handleUploadSuccess} />
        </FloatingPanel>
      </div>

      {/* flows */}
      <div
        ref={containerRef}
        className={`flows-container ${selectedFlowId ? "centered" : ""}`}
      >
        {flows
          .filter((flow) => !selectedFlowId || flow.id === selectedFlowId)
          .map((flow) => (
            <ReactFlowProvider key={flow.id}>
              <FlowCanvas
                jsonPath={flow.json_path} //"/vit_sample.json"
                canvasWidth={
                  selectedFlowId === flow.id || flows.length === 1 ? 1200 : 800
                }
                rootWidth={600}
                onClick={() => toggleFlowCanvasWidth(flow.id)}
                onDelete={() => handleDeleteFlow(flow.id)}
              />
            </ReactFlowProvider>
          ))}
      </div>
    </div>
  );
}
