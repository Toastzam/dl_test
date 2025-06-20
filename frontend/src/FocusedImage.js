// import React from 'react';

// function FocusedImage({ src, focusX, focusY, radius = 60, width = 350, height = 350 }) {
//   const size = radius * 2;
//   return (
//     <div style={{ position: 'relative', width, height }}>
//       {/* 블러 처리된 전체 이미지 */}
//       <img
//         src={src}
//         alt="blurred"
//         style={{
//           filter: 'blur(8px)',
//           width: width,
//           height: height,
//           display: 'block',
//         }}
//       />
//       {/* 선명한 원 영역 */}
//       <div
//         style={{
//           position: 'absolute',
//           left: focusX - radius,
//           top: focusY - radius,
//           width: size,
//           height: size,
//           overflow: 'hidden',
//           borderRadius: '50%',
//           pointerEvents: 'none',
//         }}
//       >
//         <img
//           src={src}
//           alt="focused"
//           style={{
//             width: width,
//             height: height,
//             position: 'absolute',
//             left: -(focusX - radius),
//             top: -(focusY - radius),
//             filter: 'none', // 블러 없음!
//           }}
//         />
//         {/* 빨간 동그라미 테두리 */}
//         <div
//           style={{
//             position: 'absolute',
//             top: 0,
//             left: 0,
//             width: size,
//             height: size,
//             border: '3px solid red',
//             borderRadius: '50%',
//             boxSizing: 'border-box',
//             pointerEvents: 'none',
//           }}
//         />
//       </div>
//     </div>
//   );
// }

// export default FocusedImage;

import React, { useRef, useEffect, useState } from 'react';

function FocusedImage({ src, focusX, focusY, radius = 60, highlight = false }) {
  const [imgSize, setImgSize] = useState({ width: 0, height: 0 });
  const imgRef = useRef();

  useEffect(() => {
    if (imgRef.current) {
      setImgSize({
        width: imgRef.current.naturalWidth,
        height: imgRef.current.naturalHeight,
      });
    }
  }, [src]);

  const size = radius * 2;

  return (
    <div style={{ position: 'relative', width: imgSize.width, height: imgSize.height }}>
      {/* 블러 처리된 전체 이미지 */}
      <img
        ref={imgRef}
        src={src}
        alt="blurred"
        style={{
          filter: 'blur(8px)',
          width: imgSize.width,
          height: imgSize.height,
          display: 'block',
        }}
        onLoad={e => {
          setImgSize({
            width: e.target.naturalWidth,
            height: e.target.naturalHeight,
          });
        }}
      />
      {/* 선명한 원 영역 */}
      <div
        style={{
          position: 'absolute',
          left: focusX - radius,
          top: focusY - radius,
          width: size,
          height: size,
          overflow: 'hidden',
          borderRadius: '50%',
          pointerEvents: 'none',
        }}
      >
        <img
          src={src}
          alt="focused"
          style={{
            width: imgSize.width,
            height: imgSize.height,
            position: 'absolute',
            left: -(focusX - radius),
            top: -(focusY - radius),
            filter: 'none',
          }}
        />
        {/* 반짝이는 테두리 */}
        <div
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: size,
            height: size,
            border: '3px solid red',
            borderRadius: '50%',
            boxSizing: 'border-box',
            pointerEvents: 'none',
            boxShadow: highlight
              ? '0 0 20px 8px rgba(255,255,0,0.7), 0 0 0 3px red'
              : '0 0 0 3px red',
            animation: highlight ? 'glow 1s infinite alternate' : 'none',
          }}
          className={highlight ? 'glow-highlight' : ''}
        />
      </div>
    </div>
  );
}

export default FocusedImage;