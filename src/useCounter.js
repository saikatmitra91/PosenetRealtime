import { useEffect, useState } from "react";

export const useCounter = () => {
  const [count, setCount] = useState(-1);

  const start = (val) => {
    setCount(val);
  }

  const reset = () => {
    setCount(-1);
  }

  useEffect(() => {
    if (count > 0) {
      setTimeout(() => {
        setCount((count) => count - 1);
      }, 1000);
    }
  }, [count]);

  return { count, start, reset };
};
