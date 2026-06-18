# BackbenchLearner – Content structure

## Folder hierarchy

- **Topic** = One folder at site root (e.g. `Backend & System Design`). Shows as a tab in the header.
- **Chapter** = Folder inside the topic (e.g. `backend`, `databases`, `caching`). Shows as a section in the sidebar.
- **Subtopic** = HTML file inside a chapter folder. Each file is one lesson/page (e.g. `what-is-api.html`).

```
BackbenchLearner/
├── index.html              ← Main app (sidebar + iframe)
├── assets/
├── styles/
├── Backend & System Design/   ← Topic
│   ├── backend/               ← Chapter
│   │   ├── what-is-api.html   ← Subtopic
│   │   ├── rest-vs-graphql.html
│   │   └── ...
│   ├── databases/
│   ├── caching/
│   ├── distributed/
│   └── reliability/
└── (future topic folder, e.g. Frontend Notes/)
```

## Adding a new topic

1. **Create the topic folder**  
   At BackbenchLearner root, create a folder with the topic name (e.g. `Frontend Notes`).

2. **Add chapters and pages**  
   Inside that folder, create one folder per chapter. Put one HTML file per subtopic in the chapter folder.  
   In each HTML file use:
   - `href="../../index.html"` for Home
   - `href="../../styles/common.css"` for the shared stylesheet

3. **Register the topic in index.html**
   - In the `TOPICS` object, add a new entry (e.g. `'frontend-notes': { ... }`).
   - Set `basePath` to the topic folder name (e.g. `'Frontend Notes'`).
   - Set `title`, `welcomeTitle`, `welcomeTagline`.
   - Set `sections` to an array of `{ title: 'Section name', links: [ { hash, path, label }, ... ] }`.
   - `path` = path relative to topic folder (e.g. `chapter-name/page.html`).
   - `hash` = same without `.html` (e.g. `chapter-name/page`), used in the URL.

4. **Add a header tab**  
   In the `<nav class="topic-nav">` in index.html, add:
   ```html
   <a href="#frontend-notes" data-topic-id="frontend-notes">Frontend Notes</a>
   ```

After that, the new topic will appear in the header and its chapters/subtopics in the sidebar when selected.
