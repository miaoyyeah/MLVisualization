import React, { useEffect } from 'react';
// @ts-ignore
import NodeGraph from './NodeGraph';
import { addEdge, useNodesState, useEdgesState, OnConnect, useReactFlow } from '@xyflow/react';

import '@xyflow/react/dist/style.css';

interface FlowCanvasProps {
  width: number | undefined;
  jsonPath?: string;
  canvasWidth?: number;
  rootWidth?: number;
  onClick?: () => void;
  onDelete?: () => void;
}

const FlowCanvas: React.FC<FlowCanvasProps> = ({
  jsonPath = '/gpt_drawing_dictionary.json',
  canvasWidth = 800,
  rootWidth = 600,
  onClick,
  onDelete,
}) => {
  const { fitView } = useReactFlow();

  // Fit view on component mount or width change
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
      <NodeGraph
        jsonPath={jsonPath}
        canvasWidth={canvasWidth}
        rootWidth={rootWidth}
      />
    </div>
  );
};

export default FlowCanvas;
