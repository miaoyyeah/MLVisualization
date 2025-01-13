import React, { useEffect, useCallback, useState } from 'react';
import {
  ReactFlow,
  Background,
  MiniMap,
  Controls,
  useReactFlow,
  Node,
  Edge,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';


interface FlowCanvasProps {
  jsonPath?: string;
  canvasWidth?: number;
  rootWidth?: number;
  onClick?: () => void;
  onDelete?: () => void;
}

interface InitialNodesData {
  nodes: Node[];
  edges: Edge[];
}

const FlowCanvas: React.FC<FlowCanvasProps> = ({
  jsonPath = '/gpt_drawing_dictionary.json',
  canvasWidth = 800,
  rootWidth = 600,
  onClick,
  onDelete,
}) => {
  const { fitView } = useReactFlow();
  const [nodes, setNodes] = useState<Node[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);
  const [selectedLabel, setSelectedLabel] = useState<string | null>(null);

  // 获取初始节点数据
  useEffect(() => {
    const fetchNodes = async () => {
      try {
        const response = await fetch(jsonPath);
        const data: InitialNodesData = await response.json();
        setNodes(data.nodes);
        setEdges(data.edges);
      } catch (error) {
        console.error('Error loading nodes:', error);
      }
    };

    fetchNodes();
  }, [jsonPath]);

  // 视图自适应
  useEffect(() => {
    const timer = setTimeout(() => {
      fitView({ padding: 0.1 });
    }, 100);

    return () => clearTimeout(timer);
  }, [canvasWidth, fitView]);

  return (
    <div className="flow-canvas" style={{ width: canvasWidth ? `${canvasWidth}px` : '800px' }} onClick={onClick}>
      <button
        className="delete-button"
        onClick={(e) => {
          e.stopPropagation();
          if (onDelete) {
            onDelete();
          }
        }}
      >
        ✕
      </button>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        fitView
      >
        <Background />
        <MiniMap />
        <Controls />
      </ReactFlow>
    </div>
  );
};

export default FlowCanvas;
