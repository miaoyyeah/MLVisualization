import React, { useEffect } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  addEdge,
  useNodesState,
  useEdgesState,
  OnConnect,
  useReactFlow,
  ReactFlowInstance,
} from '@xyflow/react';

import '@xyflow/react/dist/style.css';

interface FlowCanvasProps {
  nodes: any[];
  edges: any[];
  width: string;
  onClick?: () => void;
  onDelete?: () => void;
}

// random color
const getRandomColor = () => {
  const letters = '0123456789ABCDEF';
  let color = '#';
  for (let i = 0; i < 6; i++) {
    color += letters[Math.floor(Math.random() * 16)];
  }
  return color;
};

const FlowCanvas: React.FC<FlowCanvasProps> = ({ nodes, edges, width, onClick, onDelete }) => {
  const [nodesState, setNodes, onNodesChange] = useNodesState(
    nodes.map((node) => ({
      ...node,
      style: { backgroundColor: getRandomColor() },
    }))
  );
  const [edgesState, setEdges, onEdgesChange] = useEdgesState(edges);
  const { fitView } = useReactFlow();

  const onConnect: OnConnect = (connection) => setEdges((eds) => addEdge(connection, eds));

  // fitView
  const handleInit = (reactFlowInstance: ReactFlowInstance) => {
    reactFlowInstance.fitView({ padding: 0.1 });
  };

  useEffect(() => {
    const timer = setTimeout(() => {
      fitView({ padding: 0.1 });
    }, 100);

    return () => clearTimeout(timer);
  }, [width, fitView]);

  return (
    <div className="flow-canvas" style={{ width }} onClick={onClick}>
      <button className="delete-button" onClick={(e) => { 
        e.stopPropagation();
        if (onDelete) {
          onDelete();
        }
      }}>
        ✕
      </button>
      <ReactFlow
        nodes={nodesState}
        edges={edgesState}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onInit={handleInit}  // init fitView
      >
        <Background />
        <Controls />
      </ReactFlow>
    </div>
  );
};

export default FlowCanvas;
