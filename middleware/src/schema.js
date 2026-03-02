// schema.js – Sample GraphQL schema and resolvers (demo purposes)

const { buildSchema } = require('graphql');

// ── Type Definitions ────────────────────────────────────────────────────────
const typeDefs = `
  type User {
    id: ID!
    name: String!
    email: String!
    posts: [Post!]!
    role: String!
  }

  type Post {
    id: ID!
    title: String!
    body: String!
    author: User!
    comments: [Comment!]!
    createdAt: String!
  }

  type Comment {
    id: ID!
    text: String!
    author: User!
    post: Post!
  }

  type Query {
    users: [User!]!
    user(id: ID!): User
    posts: [Post!]!
    post(id: ID!): Post
    comments: [Comment!]!
  }
`;

// ── Seed data ───────────────────────────────────────────────────────────────
const USERS = [
    { id: '1', name: 'Alice', email: 'alice@example.com', role: 'admin' },
    { id: '2', name: 'Bob', email: 'bob@example.com', role: 'user' },
    { id: '3', name: 'Charlie', email: 'charlie@example.com', role: 'user' },
];

const POSTS = [
    { id: '1', title: 'GraphQL Security', body: 'GraphQL needs careful security…', authorId: '1', createdAt: '2024-01-01' },
    { id: '2', title: 'ML in Middleware', body: 'Isolation Forest is great for…', authorId: '2', createdAt: '2024-01-02' },
    { id: '3', title: 'EWMA Anomaly Detection', body: 'Exponential smoothing works by…', authorId: '1', createdAt: '2024-01-03' },
];

const COMMENTS = [
    { id: '1', text: 'Great post!', authorId: '2', postId: '1' },
    { id: '2', text: 'Very insightful.', authorId: '3', postId: '1' },
    { id: '3', text: 'Thanks!', authorId: '1', postId: '2' },
];

// ── Resolvers ────────────────────────────────────────────────────────────────
const resolvers = {
    Query: {
        users: () => USERS,
        user: (_, { id }) => USERS.find(u => u.id === id),
        posts: () => POSTS,
        post: (_, { id }) => POSTS.find(p => p.id === id),
        comments: () => COMMENTS,
    },
    User: {
        posts: (user) => POSTS.filter(p => p.authorId === user.id),
    },
    Post: {
        author: (post) => USERS.find(u => u.id === post.authorId),
        comments: (post) => COMMENTS.filter(c => c.postId === post.id),
    },
    Comment: {
        author: (c) => USERS.find(u => u.id === c.authorId),
        post: (c) => POSTS.find(p => p.id === c.postId),
    },
};

module.exports = { typeDefs, resolvers };
