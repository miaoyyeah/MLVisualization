import React, { useEffect } from 'react';
// @ts-ignore
import NodeGraph from './NodeGraph';
import {
  ReactFlow,
  Background,
  Controls,
  addEdge,
  useNodesState,
  useEdgesState,
  OnConnect,
  useReactFlow,
} from '@xyflow/react';

import '@xyflow/react/dist/style.css';

interface FlowCanvasProps {
  nodes: any[]; // You may want to replace `any` with a specific type
  edges: any[]; // You may want to replace `any` with a specific type
  width: string;
  jsonPath?: string;
  canvasWidth?: number;
  rootWidth?: number;
  onClick?: () => void;
  onDelete?: () => void;
}

const FlowCanvas: React.FC<FlowCanvasProps> = ({
  nodes,
  edges,
  jsonPath = '/gpt_drawing_dictionary.json',
  canvasWidth = 800,
  rootWidth = 600,
  width,
  onClick,
  onDelete,
}) => {
  const [nodesState, setNodes, onNodesChange] = useNodesState(nodes);
  const [edgesState, setEdges, onEdgesChange] = useEdgesState(edges);
  const { fitView } = useReactFlow();

  const onConnect: OnConnect = (connection) => setEdges((eds) => addEdge(connection, eds));

  // Fit view on component mount or width change
  useEffect(() => {
    const timer = setTimeout(() => {
      fitView({ padding: 0.1 });
    }, 100);

    return () => clearTimeout(timer);
  }, [width, fitView]);

  return (
    <div className="flow-canvas" style={{ width }} onClick={onClick}>
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
        nodes={nodesState}
        edges={edgesState}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        fitView
      >
        <NodeGraph jsonPath={jsonPath} canvasWidth={canvasWidth} rootWidth={rootWidth} />
        <Background />
        <Controls />
      </ReactFlow>
    </div>
  );
};

export default FlowCanvas;
