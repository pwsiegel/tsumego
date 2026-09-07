import type { GameDoc } from './data/model';

/** Download a game's SGF, injecting the display name as GN when the stored
 * SGF lacks one — the file carries all metadata for use elsewhere. Shared by
 * the game cards and the review screen, including its public view. */
export function downloadSgf(game: GameDoc): void {
  let sgf = game.sgf;
  if (game.name && !/\bGN\[/.test(sgf)) {
    sgf = sgf.replace('(;', `(;GN[${game.name.replace(/([\]\\])/g, '\\$1')}]`);
  }
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([sgf], { type: 'application/x-go-sgf' }));
  a.download = `${(game.name ?? '').replace(/[^\w\- ]+/g, '').trim() || game.id}.sgf`;
  a.click();
  URL.revokeObjectURL(a.href);
}
