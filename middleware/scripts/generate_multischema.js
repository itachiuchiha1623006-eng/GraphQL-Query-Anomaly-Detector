const { typeDefs: ecommerceTypeDefs } = require('../src/schema.js');
const { buildSchema, print } = require('graphql');
const generateQuery = require('ibm-graphql-query-generator');
const fs = require('fs');
const path = require('path');

// ── SCHEMAS ─────────────────────────────────────────────────────────────

const socialMediaTypeDefs = `
  type User {
    id: ID!
    username: String!
    bio: String
    followers(limit: Int): [User!]!
    following(limit: Int): [User!]!
    posts(limit: Int): [Post!]!
    likes: [Post!]!
    messages: [Message!]!
  }

  type Post {
    id: ID!
    author: User!
    content: String!
    mediaUrls: [String!]
    comments(limit: Int): [Comment!]!
    likes(limit: Int): [User!]!
    reposts: Int!
    createdAt: String!
  }

  type Comment {
    id: ID!
    post: Post!
    author: User!
    content: String!
    likes: Int!
    createdAt: String!
    replies(limit: Int): [Comment!]!
  }

  type Message {
    id: ID!
    sender: User!
    receiver: User!
    text: String!
    readAt: String
    sentAt: String!
  }

  type Query {
    me: User
    user(username: String!): User
    timeline(limit: Int): [Post!]!
    trending(limit: Int): [Post!]!
    searchUsers(q: String!): [User!]!
  }

  type Mutation {
    createPost(content: String!): Post
    likePost(postId: ID!): Post
    followUser(userId: ID!): User
    sendMessage(receiverId: ID!, text: String!): Message
  }
`;

const cmsTypeDefs = `
  type Page {
    id: ID!
    slug: String!
    title: String!
    blocks: [ContentBlock!]!
    seoMetadata: SEO!
    published: Boolean!
    author: Editor!
  }

  union ContentBlock = TextBlock | ImageBlock | VideoBlock | GalleryBlock

  type TextBlock { html: String! }
  type ImageBlock { url: String! caption: String }
  type VideoBlock { videoUrl: String! autoplay: Boolean! }
  type GalleryBlock { images: [ImageBlock!]! layout: String! }

  type SEO { keywords: [String!]! description: String! ogImage: String }

  type Editor {
    id: ID!
    name: String!
    articles: [Page!]!
    role: String!
  }

  type Tag {
    id: ID!
    name: String!
    pages: [Page!]!
  }

  type Query {
    page(slug: String!): Page
    allPages(limit: Int, offset: Int): [Page!]!
    tags: [Tag!]!
    tag(name: String!): Tag
    authors: [Editor!]!
  }

  type Mutation {
    publishPage(id: ID!): Page
    updateSeo(pageId: ID!, keywords: [String!]): SEO
  }
`;

const devopsTypeDefs = `
  type Organization {
    id: ID!
    name: String!
    repositories(limit: Int): [Repository!]!
    members: [Member!]!
  }

  type Member {
    id: ID!
    user: String!
    role: String!
    commits: [Commit!]!
  }

  type Repository {
    id: ID!
    name: String!
    owner: Organization!
    commits(limit: Int): [Commit!]!
    issues(state: String): [Issue!]!
    pullRequests: [PullRequest!]!
  }

  type Commit {
    hash: String!
    message: String!
    author: Member!
    repository: Repository!
    changes: Int!
  }

  type Issue {
    id: ID!
    title: String!
    body: String!
    reporter: Member!
    assignees: [Member!]!
    comments: [IssueComment!]!
  }

  type IssueComment {
    id: ID!
    author: Member!
    body: String!
  }

  type PullRequest {
    id: ID!
    title: String!
    author: Member!
    commits: [Commit!]!
    reviewers: [Member!]!
    merged: Boolean!
  }

  type Query {
    organization(name: String!): Organization
    repository(owner: String!, name: String!): Repository
    recentCommits(limit: Int): [Commit!]!
    myIssues: [Issue!]!
  }

  type Mutation {
    createIssue(repoId: ID!, title: String!, body: String!): Issue
    mergePullRequest(prId: ID!): PullRequest
  }
`;

const schemas = [
  { name: 'E-Commerce', typeDefs: ecommerceTypeDefs },
  { name: 'Social Media', typeDefs: socialMediaTypeDefs },
  { name: 'CMS Blog', typeDefs: cmsTypeDefs },
  { name: 'DevOps & Git', typeDefs: devopsTypeDefs },
];

const QUERIES_PER_SCHEMA = 5000;
const allQueries = [];

console.log(`Starting Multi-Schema Dataset Generation...`);

schemas.forEach(schemaObj => {
  console.log(`Building AST for ${schemaObj.name} schema...`);
  const schema = buildSchema(schemaObj.typeDefs);
  let successCount = 0;

  for (let i = 0; i < QUERIES_PER_SCHEMA; i++) {
    try {
      const config = {
        depthProbability: Math.random() * 0.6 + 0.1,    // 0.1 to 0.7
        breadthProbability: Math.random() * 0.6 + 0.1,  // 0.1 to 0.7
        maxDepth: Math.floor(Math.random() * 6) + 2,    // 2 to 7
        ignoreOptionalArguments: Math.random() > 0.4,
        providePlaceholders: true,
      };

      const ast = generateQuery.generateRandomQuery(schema, config);
      if (ast && ast.queryDocument) {
        allQueries.push(print(ast.queryDocument));
        successCount++;
      }
    } catch (e) {
      if (i === 0) console.error(`[Warn] Generator error on ${schemaObj.name}:`, e.message);
    }
  }
  console.log(`✅ Generated ${successCount} queries for ${schemaObj.name}.`);
});

const outPath = path.join(__dirname, '../../ml-service/data/generated_corpus.json');
fs.writeFileSync(outPath, JSON.stringify(allQueries, null, 2));

console.log(`\n🎉 Success! Wrote exactly ${allQueries.length} total universal queries to ${outPath}\n`);
