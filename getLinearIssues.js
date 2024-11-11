import { LinearClient, LinearDocument } from '@linear/sdk';
import { promises as fs } from 'fs';

const linearClient = new LinearClient({
  apiKey: process.env.LINEAR_API_KEY
});

async function getProjectIssues(projectId, options = {}) {
  try {
    const project = await linearClient.project(projectId);

    if (!project) {
      throw new Error('Project not found');
    }


    // Get filtered issues as before
    const issues = await project.issues({
      first: options.limit || 50,
      orderBy: options.orderBy || LinearDocument.PaginationOrderBy.UpdatedAt,
      includeArchived: options.includeArchived || false,
      filter: options.filter
    });

    // Get comments for each issue
    const issuesWithComments = await Promise.all(issues.nodes.map(async (issue) => {
      const comments = await issue.comments();
      return {
        ...issue,
        comments: comments.nodes
      };
    }));

    return {
      success: true,
      issues: issuesWithComments,
      pageInfo: issues.pageInfo,
    };

  } catch (error) {
    console.error('Error fetching project issues:', error);
    return {
      success: false,
      error: error.message,
      issues: [],
    };
  }
}

// Separate function to format the output
function formatIssueOutput(issueData) {
  const formattedIssues = issueData.issues.map(issue => ({
    id: issue.id,
    identifier: issue.identifier,
    title: issue.title,
    description: issue.description,
    createdAt: issue.createdAt,
    updatedAt: issue.updatedAt,
    completedAt: issue.completedAt,
    url: issue.url,
    comments: issue.comments.map(comment => ({
      id: comment.id,
      body: comment.body,
      createdAt: comment.createdAt,
      updatedAt: comment.updatedAt
    }))
  }));

  return {
    success: issueData.success,
    issues: formattedIssues,
    pageInfo: issueData.pageInfo,
  };
}
// Alternative formatting function for a more concise output
function formatIssuesSummary(issueData) {
  return issueData.issues.map(issue => ({
    identifier: issue.identifier,
    title: issue.title,
    commentCount: issue.comments.length
  }));
}

// Usage
async function main() {
  const result = await getProjectIssues('cd06d107f422', {
    limit: 200,
    orderBy: LinearDocument.PaginationOrderBy.CreatedAt,
    includeArchived: true,
    // filter: {
    //   state: { name: { eq: 'Deployed' } }
    // }
  });

  const formattedResult = formatIssueOutput(result);
  const summaryOutput = formatIssuesSummary(result);

  // Save the full formatted result
  await fs.writeFile(
    'linear-issues.json', 
    JSON.stringify(formattedResult, null, 2)
  );

  // Optionally, save the summary as well
  await fs.writeFile(
    'linear-issues-summary.json', 
    JSON.stringify(summaryOutput, null, 2)
  );

  console.log('Results saved to linear-issues.json');
}

main().catch(console.error);