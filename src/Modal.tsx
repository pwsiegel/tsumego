import { useEffect, type ReactNode } from 'react';
import { useBackdropDismiss } from './backdrop';
import './Modal.css';

/** Centered modal dialog over the current view. Closes on backdrop / × / Escape.
 * Escape is captured (and its propagation stopped) so this dialog closes first
 * when it's nested inside another modal — e.g. Explore settings over the problem
 * modal — instead of the outer one closing. */
export function Modal({
  open, onClose, title, className, children,
}: {
  open: boolean;
  onClose: () => void;
  title?: string;
  className?: string;
  children: ReactNode;
}) {
  const backdrop = useBackdropDismiss(onClose);
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { e.stopPropagation(); onClose(); }
    };
    document.addEventListener('keydown', onKey, true);
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      document.removeEventListener('keydown', onKey, true);
      document.body.style.overflow = prevOverflow;
    };
  }, [open, onClose]);

  if (!open) return null;
  return (
    <div className="modal-backdrop" {...backdrop} role="presentation">
      <div
        className={className ? `modal-card ${className}` : 'modal-card'}
        role="dialog"
        aria-modal="true"
      >
        <div className="modal-head">
          {title && <span className="modal-title">{title}</span>}
          <button type="button" className="modal-close" onClick={onClose} aria-label="Close">×</button>
        </div>
        <div className="modal-body">{children}</div>
      </div>
    </div>
  );
}
