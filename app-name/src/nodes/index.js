import { SyncLabelNode } from "./SyncLabelNode";

// Recursion
const processNode = (node, parentId = null) => {
  const SCALE = 5;
  const nodes = [];
  
  const flowNode = {
    id: node.id,
    type: node.label === "Root" ? "rootNode" : "default",
    data: { label: node.label, level: node.level },
    position: { x: node.x * SCALE, y: node.y * SCALE  },
    style: {
      background: node.color,
      width: node.width * SCALE,
      height: node.height * SCALE,
      border: node.label === "Root" ? "none" : undefined,
      fontSize: "16px",
    },
    // parentId: parentId,
    // extent: "parent",
  };
  
  nodes.push(flowNode);

  if (node.children) {
    if (Array.isArray(node.children)) {
      node.children.forEach(child => {
        const childNodes = processNode(child, node.id);
        nodes.push(...childNodes);
      });
    } else if (typeof node.children === 'object') {
      const childNodes = processNode(node.children, node.id);
      nodes.push(...childNodes);
    }
  }

  return nodes;
};

// 新增生成edges的函数
const generateEdges = (nodes) => {
  // 仅按level对节点进行分组
  const nodesByLevel = {};
  nodes.forEach(node => {
    const level = node.data.level;
    if (!nodesByLevel[level]) {
      nodesByLevel[level] = [];
    }
    nodesByLevel[level].push(node);
  });

  const edges = [];
  Object.values(nodesByLevel).forEach(levelNodes => {
    // 按id排序
    levelNodes.sort((a, b) => a.id.localeCompare(b.id));
    
    // 连接相邻且id倒数第二个字符相同的节点
    for (let i = 0; i < levelNodes.length - 1; i++) {
      const currentId = levelNodes[i].id;
      const nextId = levelNodes[i + 1].id;
      const currentChar = currentId.charAt(currentId.length - 3);
      const nextChar = nextId.charAt(nextId.length - 3);
      
      if (currentChar === nextChar) {
        edges.push({
          id: `e${currentId}-${nextId}`,
          source: currentId,
          target: nextId,
          type: 'smoothstep',
          markerEnd: { type: 'arrowclosed'},
          style: {
            strokeWidth: 2,
          }
        });
      }
    }
  });
  
  return edges;
};

export const getInitialNodes = (data) => {
  try {
    const nodes = processNode(data.sublayers[0].children);
    const edges = generateEdges(nodes);
    return { nodes, edges };
  } catch (error) {
    console.error('Error loading nodes:', error);
    return [];
  }
};


export const nodeTypes = {
  "sync-label": SyncLabelNode,
  // "rootNode": RootNode,
};



// import { PositionLoggerNode } from "./PositionLoggerNode";
// import { AttentionColorNode } from "./AttentionColor";
// import { SyncLabelNode } from "./SyncLabelNode";


// export const initialNodes = [
//   {
//     id: 'A',
//     type: 'default',
//     data: { label: null },
//     position: { x: 0, y: 0 },
//     style: {
//       width: 170,
//       height: 140,
//     },
//   },
//   {
//     id: 'B',
//     type: "position-logger",
//     data: { label: 'child node 1' },
//     position: { x: 10, y: 10 },
//     parentId: 'A',
//     extent: 'parent',
//   },
//   {
//     id: 'C',
//     type: "position-logger",
//     data: { label: 'child node 1' },
//     position: { x: 10, y: 90 },
//     parentId: 'A',
//     extent: 'parent',
//   },
//   {
//     id: 'D',
//     type: 'default',
//     data: { label: 'parent node' },
//     position: { x: 0, y: 170 },
//     style: {
//       width: 170,
//       height: 140,
//     },
//   },
// ];

// export const nodeTypes = {
//   "position-logger": PositionLoggerNode,
//   // Add any of your custom nodes here!
//   "attention-color": AttentionColorNode,
//   "sync-label": SyncLabelNode,
// };