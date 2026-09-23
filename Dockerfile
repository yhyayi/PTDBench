FROM busybox AS b
FROM scratch AS emptystage
FROM b AS t1
COPY --from=ctx verl/ /dest/
RUN ls -la /dest
FROM b AS t2
COPY --from=ctx missing/ /dest/
RUN ls -la /dest
FROM b AS t3
COPY --from=emptystage / /dest/
RUN ls -la /dest && echo "T3-OK"
FROM b AS t4
COPY --from=ctx . /dest/
RUN ls -laR /dest && echo "T4-OK"
