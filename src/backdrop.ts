import { useRef, type MouseEvent } from 'react';

/** Backdrop props that dismiss a dialog on a click of the backdrop itself.
 * A click is delivered to the nearest ancestor holding both the press and the
 * release, so selecting text inside a dialog and releasing outside it lands a
 * click on the backdrop; requiring the press to have started there keeps that
 * from throwing the dialog away mid-edit. */
export function useBackdropDismiss(onDismiss: () => void) {
  const startedOnBackdrop = useRef(false);
  return {
    onMouseDown: (e: MouseEvent) => { startedOnBackdrop.current = e.target === e.currentTarget; },
    onClick: (e: MouseEvent) => {
      if (startedOnBackdrop.current && e.target === e.currentTarget) onDismiss();
    },
  };
}
