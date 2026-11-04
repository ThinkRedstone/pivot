import { defineRouteMiddleware } from "@astrojs/starlight/route-data";

export const onRequest = defineRouteMiddleware((context) => {
  const { toc, entry } = context.locals.starlightRoute;

  if (toc) {
    // Gather the sections, including any nested under Starlight's Overview,
    // then put them all beneath a single page-title entry.
    const sections = toc.items.flatMap((item) =>
      item.slug === "_top" ? item.children : [item],
    );
    toc.items = [
      { depth: 1, slug: "_top", text: entry.data.title, children: sections },
    ];
  }
});
