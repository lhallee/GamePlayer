# Go Rules Used By Game Player

This implementation uses a compact automated ruleset suitable for self-play.

## Board

Go is played on a square grid. The default board size is 19x19. Tests and smoke
runs may use smaller boards.

## Turns

Black moves first. Players alternate turns. A turn is either:

- placing a stone on an empty point
- passing

## Groups And Liberties

Orthogonally adjacent stones of the same color form a group. Empty
orthogonally adjacent points are liberties.

After a move, any opposing group with no liberties is removed. Then any of the
mover's own groups with no liberties are removed. This follows Tromp-Taylor and
means suicide is legal.

## Repetition

Moves that leave a grid pattern identical to one that the mover has previously
left are illegal. Passes also count as positions left by the passing player.

## End Of Game

The game ends after two consecutive passes.

## Scoring

The implementation uses area scoring:

- one point for each live stone on the board
- one point for each empty point enclosed only by that player
- komi is added to White

The default komi is 7.5 points.
