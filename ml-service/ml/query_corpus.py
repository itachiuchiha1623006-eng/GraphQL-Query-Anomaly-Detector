"""
query_corpus.py
================
A curated corpus of realistic GraphQL query strings representing the true
distribution of legitimate traffic. Used to train the model on real query
patterns rather than only synthetic numeric vectors.

Categories:
  - E-commerce   (products, orders, carts, reviews)
  - User / Auth  (profiles, sessions, permissions)
  - Social       (posts, comments, follows, feeds)
  - Blog / CMS   (articles, tags, categories)
  - Analytics    (metrics, events, dashboards)

Each query is a valid GraphQL document string.

Usage:
  from ml.query_corpus import NORMAL_QUERIES
  # → list of 200+ query strings
"""

# ── E-commerce queries ──────────────────────────────────────────────────────

ECOMMERCE_QUERIES = [
    # --- Products ---
    "query { products { id name price } }",
    "query { product(id: \"1\") { id name description price stock } }",
    "query { products(category: \"electronics\") { id name price thumbnail } }",
    "query { products(limit: 10, offset: 0) { id name price rating } }",
    """query GetProduct($id: ID!) {
  product(id: $id) {
    id
    name
    description
    price
    stock
    category {
      id
      name
    }
  }
}""",
    """query SearchProducts($term: String!, $limit: Int) {
  searchProducts(query: $term, limit: $limit) {
    id
    name
    price
    thumbnail
    rating
  }
}""",
    """query ProductWithReviews($id: ID!) {
  product(id: $id) {
    id
    name
    price
    reviews {
      id
      rating
      comment
      author {
        id
        name
      }
    }
  }
}""",
    """query FeaturedProducts {
  featuredProducts {
    id
    name
    price
    discount
    thumbnail
  }
}""",
    "query { categories { id name slug productCount } }",
    "query { category(slug: \"shoes\") { id name description products { id name price } } }",

    # --- Orders ---
    "query { orders { id status total createdAt } }",
    "query { order(id: \"ORD-42\") { id status total items { product { name } quantity } } }",
    """query MyOrders($userId: ID!) {
  orders(userId: $userId) {
    id
    status
    total
    createdAt
    updatedAt
  }
}""",
    """query OrderDetails($id: ID!) {
  order(id: $id) {
    id
    status
    total
    shippingAddress {
      street
      city
      country
    }
    items {
      product {
        id
        name
        price
      }
      quantity
      unitPrice
    }
  }
}""",
    "query { recentOrders(limit: 5) { id status total createdAt } }",

    # --- Cart ---
    "query { cart { id items { product { id name price } quantity } total } }",
    "query { cartItemCount }",
    """query CartSummary($userId: ID!) {
  cart(userId: $userId) {
    id
    total
    itemCount
    items {
      id
      quantity
      product {
        name
        price
        thumbnail
      }
    }
  }
}""",

    # --- Reviews ---
    "query { reviews(productId: \"1\") { id rating comment author { name } } }",
    """query ProductReviews($productId: ID!, $limit: Int) {
  reviews(productId: $productId, limit: $limit) {
    id
    rating
    comment
    createdAt
    author {
      id
      name
      avatar
    }
  }
}""",

    # --- Mutations ---
    """mutation AddToCart($productId: ID!, $quantity: Int!) {
  addToCart(productId: $productId, quantity: $quantity) {
    id
    total
    itemCount
  }
}""",
    """mutation PlaceOrder($input: OrderInput!) {
  placeOrder(input: $input) {
    id
    status
    total
  }
}""",
    """mutation CreateReview($productId: ID!, $rating: Int!, $comment: String) {
  createReview(productId: $productId, rating: $rating, comment: $comment) {
    id
    rating
    comment
  }
}""",
    """mutation UpdateCartItem($itemId: ID!, $quantity: Int!) {
  updateCartItem(itemId: $itemId, quantity: $quantity) {
    id
    quantity
    total
  }
}""",
    "mutation { removeFromCart(itemId: \"item-5\") { id itemCount } }",
]

# ── User / Auth queries ─────────────────────────────────────────────────────

USER_AUTH_QUERIES = [
    "query { me { id name email createdAt } }",
    "query { user(id: \"1\") { id name email avatar bio } }",
    "query { users(limit: 20) { id name email } }",
    """query UserProfile($id: ID!) {
  user(id: $id) {
    id
    name
    email
    avatar
    bio
    createdAt
  }
}""",
    """query Me {
  me {
    id
    name
    email
    role
    permissions
  }
}""",
    """mutation Login($email: String!, $password: String!) {
  login(email: $email, password: $password) {
    token
    user {
      id
      name
      email
    }
  }
}""",
    """mutation Register($input: RegisterInput!) {
  register(input: $input) {
    token
    user {
      id
      name
      email
    }
  }
}""",
    """mutation UpdateProfile($input: ProfileInput!) {
  updateProfile(input: $input) {
    id
    name
    bio
    avatar
  }
}""",
    "mutation { logout { success } }",
    """mutation ChangePassword($old: String!, $new: String!) {
  changePassword(oldPassword: $old, newPassword: $new) {
    success
    message
  }
}""",
    """mutation RequestPasswordReset($email: String!) {
  requestPasswordReset(email: $email) {
    success
  }
}""",
    "query { roles { id name permissions } }",
    "query { userPermissions { resource action } }",
    """query SearchUsers($query: String!) {
  searchUsers(query: $query) {
    id
    name
    email
    avatar
  }
}""",
    "query { userCount }",
    """query UserActivity($userId: ID!) {
  userActivity(userId: $userId) {
    lastLogin
    totalOrders
    totalReviews
  }
}""",
]

# ── Social / Feed queries ───────────────────────────────────────────────────

SOCIAL_QUERIES = [
    "query { feed { id content author { name } createdAt } }",
    "query { post(id: \"p1\") { id content likes commentsCount } }",
    """query Feed($limit: Int, $cursor: String) {
  feed(limit: $limit, cursor: $cursor) {
    pageInfo {
      hasNextPage
      endCursor
    }
    edges {
      node {
        id
        content
        createdAt
        author {
          id
          name
          avatar
        }
        likesCount
      }
    }
  }
}""",
    """query PostDetails($id: ID!) {
  post(id: $id) {
    id
    content
    createdAt
    author {
      id
      name
    }
    likesCount
    commentsCount
    tags
  }
}""",
    "query { comments(postId: \"p1\") { id content author { name } } }",
    """query Comments($postId: ID!, $limit: Int) {
  comments(postId: $postId, limit: $limit) {
    id
    content
    createdAt
    author {
      id
      name
      avatar
    }
    likesCount
  }
}""",
    """mutation CreatePost($content: String!, $tags: [String]) {
  createPost(content: $content, tags: $tags) {
    id
    content
    createdAt
  }
}""",
    """mutation LikePost($postId: ID!) {
  likePost(postId: $postId) {
    id
    likesCount
    isLiked
  }
}""",
    "mutation { followUser(userId: \"u2\") { id followers } }",
    "mutation { unfollowUser(userId: \"u2\") { success } }",
    "query { followers(userId: \"u1\") { id name avatar } }",
    "query { following(userId: \"u1\") { id name avatar } }",
    """query Notifications($limit: Int) {
  notifications(limit: $limit) {
    id
    type
    message
    read
    createdAt
  }
}""",
    "mutation { markNotificationsRead { success count } }",
    """mutation AddComment($postId: ID!, $content: String!) {
  addComment(postId: $postId, content: $content) {
    id
    content
    createdAt
  }
}""",
    "query { trendingPosts(limit: 10) { id content likesCount } }",
    "query { suggestedUsers(limit: 5) { id name avatar mutualFollowers } }",
]

# ── Blog / CMS queries ──────────────────────────────────────────────────────

BLOG_QUERIES = [
    "query { articles { id title slug publishedAt } }",
    "query { article(slug: \"hello-world\") { id title content publishedAt } }",
    """query ArticleList($limit: Int, $page: Int) {
  articles(limit: $limit, page: $page) {
    id
    title
    slug
    excerpt
    publishedAt
    author {
      id
      name
    }
    tags {
      id
      name
    }
  }
}""",
    """query ArticleDetail($slug: String!) {
  article(slug: $slug) {
    id
    title
    content
    publishedAt
    author {
      id
      name
      bio
      avatar
    }
    tags {
      id
      name
      slug
    }
  }
}""",
    "query { tags { id name slug articleCount } }",
    "query { tag(slug: \"graphql\") { id name articles { id title slug } } }",
    "query { categories { id name slug description } }",
    """query ArticlesByTag($tag: String!) {
  articlesByTag(tag: $tag) {
    id
    title
    slug
    excerpt
    publishedAt
  }
}""",
    """mutation CreateArticle($input: ArticleInput!) {
  createArticle(input: $input) {
    id
    title
    slug
    publishedAt
  }
}""",
    """mutation UpdateArticle($id: ID!, $input: ArticleInput!) {
  updateArticle(id: $id, input: $input) {
    id
    title
    updatedAt
  }
}""",
    "mutation { publishArticle(id: \"a1\") { id publishedAt } }",
    "mutation { unpublishArticle(id: \"a1\") { id status } }",
    "query { draftArticles { id title updatedAt } }",
    "query { featuredArticles(limit: 3) { id title slug thumbnail } }",
    """query SearchArticles($query: String!) {
  searchArticles(query: $query) {
    id
    title
    excerpt
    slug
  }
}""",
    "query { relatedArticles(articleId: \"a1\", limit: 4) { id title slug } }",
]

# ── Analytics queries ───────────────────────────────────────────────────────

ANALYTICS_QUERIES = [
    "query { pageViews(period: \"7d\") { date count } }",
    "query { totalRevenue(period: \"30d\") }",
    """query DashboardMetrics($period: String!) {
  metrics(period: $period) {
    totalUsers
    newUsers
    totalOrders
    totalRevenue
    averageOrderValue
  }
}""",
    """query TopProducts($limit: Int, $period: String) {
  topProducts(limit: $limit, period: $period) {
    product {
      id
      name
    }
    sales
    revenue
  }
}""",
    "query { conversionRate(period: \"30d\") { rate change } }",
    """query UserGrowth($from: String!, $to: String!) {
  userGrowth(from: $from, to: $to) {
    date
    newUsers
    totalUsers
  }
}""",
    "query { topCategories(limit: 5) { category { name } revenue } }",
    """query RevenueByPeriod($granularity: String!) {
  revenueByPeriod(granularity: $granularity) {
    period
    revenue
    orders
  }
}""",
    "query { sessionCount(period: \"24h\") }",
    "query { bounceRate(period: \"7d\") }",
    """query EventLog($event: String!, $limit: Int) {
  events(type: $event, limit: $limit) {
    id
    type
    userId
    properties
    createdAt
  }
}""",
    """query FunnelAnalysis($steps: [String!]!) {
  funnel(steps: $steps) {
    step
    users
    conversionRate
  }
}""",
    "query { retentionRate(cohort: \"2024-01\") { week rate } }",
    "query { activeUsers(period: \"1d\") }",
    "query { averageSessionDuration(period: \"7d\") }",
]

# ── Fragment queries ────────────────────────────────────────────────────────

FRAGMENT_QUERIES = [
    """fragment UserFields on User {
  id
  name
  email
  avatar
}

query { me { ...UserFields } }""",
    """fragment ProductCore on Product {
  id
  name
  price
  thumbnail
}

query { products(limit: 10) { ...ProductCore } }""",
    """fragment ArticleSummary on Article {
  id
  title
  slug
  excerpt
  publishedAt
}

query { articles { ...ArticleSummary } }""",
    """fragment PageInfo on PageInfo {
  hasNextPage
  hasPreviousPage
  startCursor
  endCursor
}

query Feed($cursor: String) {
  feed(cursor: $cursor) {
    pageInfo { ...PageInfo }
    edges { node { id content } }
  }
}""",
    """fragment AuthorFields on User {
  id
  name
  avatar
  bio
}

query PostWithAuthor($id: ID!) {
  post(id: $id) {
    id
    content
    author { ...AuthorFields }
  }
}""",
]

# ── Simple / Minimal queries ────────────────────────────────────────────────

SIMPLE_QUERIES = [
    "{ __typename }",
    "query { me { id } }",
    "query Ping { ping }",
    "query { version }",
    "query { serverTime }",
    "query { config { maxUploadSize allowedTypes } }",
    "query { healthCheck { status } }",
    "query { countries { code name } }",
    "query { currencies { code symbol } }",
    "query { timezones { name offset } }",
    "{ users { id name } }",
    "{ products { id name price } }",
    "{ posts { id title } }",
    "query { me { name email } }",
    "query { settings { theme language notifications } }",
    "query { recentActivity { type createdAt } }",
    "query { unreadCount }",
    "query { featuredBrands { id name logo } }",
    "query { banners { id image link } }",
    "query { faqs { question answer } }",
]

# ── Mutation samples ────────────────────────────────────────────────────────

MUTATION_QUERIES = [
    "mutation { refreshToken { token expiresAt } }",
    """mutation UpdateSettings($input: SettingsInput!) {
  updateSettings(input: $input) {
    theme
    language
    notifications
  }
}""",
    """mutation UploadAvatar($file: Upload!) {
  uploadAvatar(file: $file) {
    url
  }
}""",
    "mutation { deleteAccount { success } }",
    """mutation SendMessage($to: ID!, $content: String!) {
  sendMessage(to: $to, content: $content) {
    id
    content
    createdAt
  }
}""",
    """mutation MarkAsRead($messageId: ID!) {
  markAsRead(messageId: $messageId) {
    id
    read
  }
}""",
    "mutation { clearCart { success } }",
    """mutation ApplyCoupon($code: String!) {
  applyCoupon(code: $code) {
    discount
    total
  }
}""",
    """mutation CancelOrder($orderId: ID!, $reason: String) {
  cancelOrder(orderId: $orderId, reason: $reason) {
    id
    status
  }
}""",
    """mutation TrackEvent($event: String!, $properties: JSON) {
  trackEvent(event: $event, properties: $properties) {
    id
  }
}""",
]

# ── Combined corpus ─────────────────────────────────────────────────────────

NORMAL_QUERIES: list[str] = (
    ECOMMERCE_QUERIES
    + USER_AUTH_QUERIES
    + SOCIAL_QUERIES
    + BLOG_QUERIES
    + ANALYTICS_QUERIES
    + FRAGMENT_QUERIES
    + SIMPLE_QUERIES
    + MUTATION_QUERIES
)

QUERY_CATEGORIES = {
    "ecommerce":  ECOMMERCE_QUERIES,
    "user_auth":  USER_AUTH_QUERIES,
    "social":     SOCIAL_QUERIES,
    "blog":       BLOG_QUERIES,
    "analytics":  ANALYTICS_QUERIES,
    "fragments":  FRAGMENT_QUERIES,
    "simple":     SIMPLE_QUERIES,
    "mutations":  MUTATION_QUERIES,
}

if __name__ == "__main__":
    print(f"Total corpus size: {len(NORMAL_QUERIES)} queries")
    for cat, queries in QUERY_CATEGORIES.items():
        print(f"  {cat:<12}: {len(queries)} queries")
