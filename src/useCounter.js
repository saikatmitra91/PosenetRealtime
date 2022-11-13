import { useEffect, useState } from "react";

export const useCounter = () => {
  const [counter, setCounter] = useState(-1);

  const startCounter = (val) => {
    setCounter(val);
  }

  useEffect(() => {
    if (counter >= 0) {
      setTimeout(() => {
        setCounter((count) => count - 1);
      }, 1000);
    }
  }, [counter]);

  return {  counter, startCounter };
};
