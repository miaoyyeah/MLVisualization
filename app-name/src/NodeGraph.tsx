import React, { useState, useEffect } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  Edge,
  Node,
  BackgroundVariant,
  MarkerType,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

interface NodeGraphProps {
  jsonPath?: string;
  canvasWidth?: number;
  rootWidth?: number;
  onDelete?: () => void; // 新增删除事件的回调
  onClick?: () => void;  // 新增点击事件的回调
}

interface Layer {
  id: string;
  label: string;
  layout?: 'horizontal' | 'vertical';
  color?: string;
  children?: Layer[];
}

const NodeGraph: React.FC<NodeGraphProps> = ({
  jsonPath = '/drawing_dictionary.json',
  canvasWidth = 800,
  rootWidth = 600,
  onDelete,
  onClick,
}) => {
  const [nodes, setNodes] = useState<Node[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);

  useEffect(() => {
    fetch(jsonPath)
      .then((response) => {
        if (!response.ok) {
          throw new Error('Failed to fetch JSON file');
        }
        return response.json();
      })
      .then((data) => {
        console.log('Loaded JSON:', data);
        const { layers } = data;
        const generatedNodes: Node[] = [];
        const generatedEdges: Edge[] = [];
        let yPosition = 0;

        const processLayer = (layer: Layer, parentId: string | null = null, parentWidth = rootWidth, level = 1) => {
          const layerId = layer.id;
          let oldYPosition = yPosition;

          yPosition += 50; // Spacing between nodes

          let currentWidth = parentWidth - 50;
          if (layer.layout === 'horizontal') {
            currentWidth = currentWidth / 4;
          }

          if (layer.children) {
            const subLayerIds: string[] = [];
            let childTotalWidth = 0;

            layer.children.forEach((child, index) => {
              const childWidth = currentWidth - 50;
              const childXOffset = canvasWidth / 2 - currentWidth / 2 + childTotalWidth;
              childTotalWidth += childWidth + 50;

              processLayer(child, layerId, currentWidth, level + 1);
              subLayerIds.push(child.id);
            });

            if (layer.label !== 'Multihead Attention') {
              for (let i = 0; i < subLayerIds.length - 1; i++) {
                generatedEdges.push({
                  id: `edge-${subLayerIds[i]}-${subLayerIds[i + 1]}`,
                  source: subLayerIds[i],
                  target: subLayerIds[i + 1],
                  type: 'straight',
                  markerEnd: { type: MarkerType.ArrowClosed },
                  style: {
                    strokeWidth: layer.layout === 'horizontal' ? 0 : 3,
                  },
                });
              }
            }
          }

          let xOffset = canvasWidth / 2 - currentWidth / 2;
          let new_xOffset;

          if (layer.layout === 'horizontal') {
            new_xOffset =
              xOffset +
              (Number(layer.id.slice(-1)) - 1) * currentWidth * 1.5 -
              (2 * currentWidth - 37.5);
            oldYPosition = oldYPosition - (Number(layer.id.slice(-1)) - 1) * 50;
          } else {
            new_xOffset = xOffset;
            yPosition += 50;
          }

          const node: Node = {
            id: layerId,
            position: { x: new_xOffset, y: oldYPosition },
            data: { label: layer.label },
            style: {
              width: currentWidth,
              ...(layer.layout !== 'horizontal'
                ? { height: yPosition - oldYPosition - 50 }
                : { height: 100 }),
              backgroundColor: layer.color,
              fontSize: '20px',
              ...(level > 1 ? { border: '2px solid #0077cc' } : { border: '0px' }),
            },
          };
          generatedNodes.push(node);
        };

        layers.forEach((layer: Layer) => processLayer(layer));
        setNodes(generatedNodes);
        setEdges(generatedEdges);
      })
      .catch((error) => console.error('Error loading JSON:', error));
  }, [jsonPath, canvasWidth, rootWidth]);

  return (
    <div
      style={{ width: '100%', height: '100%' }}
      onClick={onClick} // 将点击事件绑定到容器
    >
      <button
        className="delete-button"
        onClick={(e) => {
          e.stopPropagation(); // 防止事件冒泡触发容器的 onClick
          if (onDelete) {
            onDelete();
          }
        }}
      >
        ✕
      </button>
      <ReactFlow nodes={nodes} edges={edges} fitView>
        <Background variant={BackgroundVariant.Dots} gap={16} size={1} />
        <Controls />
      </ReactFlow>
    </div>
  );
};

export default NodeGraph;
